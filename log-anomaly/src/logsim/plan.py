"""Everything a run does beyond steady traffic, decided before any record exists.

The plan holds the natural episodes (bursts, maintenance windows, exports,
daily batches, retry echoes, deploys), the injected scenarios, and the ground
truth labelling both. The record generator
and the truth file read this one plan, so a labelled window and the records
inside it cannot disagree.

Scenarios come in matched pairs: an injected case that must alert reproduces
the raw signal of a natural case that must stay quiet, drawn from the same
distribution, so only an entity's own history or its metadata separates them.
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta
from zoneinfo import ZoneInfo

import numpy as np

from logsim.config import Config, MessageTemplate, Range
from logsim.kinds import SIGNALS
from logsim.repro import Stream, rng, zone
from logsim.traffic import profile
from logsim.world import Key, Org, World

NS = 10**9
MINUTE = 60 * NS
HOUR = 60 * MINUTE
DAY = 24 * HOUR

# Draws per scenario before the schedule is declared unplaceable.
_ATTEMPTS_PER_SCENARIO = 1000
# A provider outage hitting one org is an unplanned_outage; this kind is about
# many orgs failing at once.
_PROVIDER_OUTAGE_MIN_ORGS = 2
# An echo has ended once its expected retries per second fall below this.
_ECHO_DONE_INTENSITY = 0.01
_ECHO_MAX_SECONDS = 3600
# How far an alert case keeps from a natural quiet case it shares a signal
# with. Wider than the scorer's default deadline, so a late page on one cannot
# land on the other.
_QUIET_MARGIN = 30 * MINUTE
SERVER_ERROR = 500
OVERLOADED = 503


class PlanError(ValueError):
    pass


@dataclass(frozen=True)
class Burst:
    key: int
    start: int
    end: int
    rate: float  # requests/s before rate_scale


@dataclass(frozen=True)
class Outage:
    org: int
    start: int
    end: int


@dataclass(frozen=True)
class Silence:
    key: int
    start: int
    end: int


@dataclass(frozen=True)
class Export:
    key: int
    id: str
    parts: tuple[tuple[int, int], ...]  # (time ns, bytes)


@dataclass(frozen=True)
class NovelError:
    key: int
    time: int
    template: MessageTemplate


@dataclass(frozen=True)
class ErrorSpan:
    """A key's requests fail with `status` at `share` over [start, end)."""

    key: int
    start: int
    end: int
    share: float
    status: int


@dataclass(frozen=True)
class Surge:
    """A key's request rate is multiplied by `multiplier` over [start, end)."""

    key: int
    start: int
    end: int
    multiplier: float


@dataclass(frozen=True)
class Ramp:
    """A multiplier rising linearly from 1 at `start` to `factor` at `end`. A
    ramp that holds stays at `factor` afterwards; one that does not drops back
    to 1, as a leaking process does when it restarts."""

    org: int
    start: int
    end: int
    factor: float
    holds: bool


@dataclass(frozen=True)
class Cascade:
    """Retries one key's clients send after a dependency fails at `start`.
    Retries fail before `overload_end` and succeed after."""

    key: int
    start: int
    overload_end: int
    attempts: tuple[int, ...]  # retries sent in each second from start


@dataclass(frozen=True)
class Deploy:
    time: int
    release: str


@dataclass(frozen=True)
class RenameAt:
    """From `time` on, background template `template` is logged as `to`."""

    time: int
    template: int
    to: MessageTemplate


@dataclass(frozen=True)
class Activation:
    start: int  # no records for the org before this
    ramp_end: int  # traffic grows linearly from zero until this


@dataclass(frozen=True)
class Truth:
    id: str
    kind: str
    pair: str  # matched-pair family the case belongs to
    expect: str  # "alert" or "quiet"
    org: int | None  # None for fleet-wide cases, which list their entities in detail
    key: int | None
    start: int
    end: int
    scored: bool  # starts inside the scored phase
    detail: dict[str, object]


@dataclass(frozen=True)
class Plan:
    start: int
    scored_from: int
    end: int
    bursts: dict[int, tuple[Burst, ...]]
    silences: dict[int, tuple[Silence, ...]]
    outages: dict[int, tuple[Outage, ...]]
    exports: tuple[Export, ...]
    novel_errors: tuple[NovelError, ...]
    activations: dict[int, Activation]
    surges: dict[int, tuple[Surge, ...]]  # by key
    error_spans: dict[int, tuple[ErrorSpan, ...]]  # by key
    rate_ramps: dict[int, tuple[Ramp, ...]]  # by org
    latency_ramps: dict[int, tuple[Ramp, ...]]  # by org
    cascades: tuple[Cascade, ...]
    deploys: tuple[Deploy, ...]
    renames: tuple[RenameAt, ...]
    truth: tuple[Truth, ...]

    @property
    def chunks(self) -> int:
        return (self.end - self.start) // HOUR

    def active(self, org: int, at: int) -> bool:
        return _active(self.activations, org, at)


def build_plan(cfg: Config, world: World, seed: int, start: int) -> Plan:
    if start % HOUR:
        # Hour chunks must align with UTC hours: every DST transition in the
        # pinned zones lands on one, so a chunk never straddles an offset change.
        raise PlanError("start must fall exactly on a UTC hour")
    return _Builder(cfg, world, seed, start).build()


@dataclass
class _Window:
    org: int
    start: int
    end: int
    detail: dict[str, object]
    overrun: bool = False


@dataclass
class _Echo:
    key: int
    start: int
    overload_end: int
    attempts: tuple[int, ...]

    @property
    def end(self) -> int:
        return max(self.overload_end, self.start + len(self.attempts) * NS)


@dataclass
class _Pending:
    kind: str
    pair: str
    expect: str
    org: int | None  # None for fleet-wide cases
    key: int | None
    start: int
    end: int
    detail: dict[str, object] = field(default_factory=dict)


class _Builder:
    def __init__(self, cfg: Config, world: World, seed: int, start: int):
        self.cfg = cfg
        self.world = world
        self.seed = seed
        self.start = start
        self.scored_from = start + cfg.warmup_days * DAY
        self.end = self.scored_from + cfg.scored_days * DAY
        self.scale = cfg.traffic.rate_scale
        self.gap = int(cfg.scenarios.min_gap_hours * HOUR)

        self.natural_bursts: list[Burst] = []
        self.windows: list[_Window] = []
        self.ad_hoc_exports: list[Export] = []
        self.weekly_exports: list[Export] = []

        self.bursts: dict[int, list[Burst]] = defaultdict(list)
        self.silences: dict[int, list[Silence]] = defaultdict(list)
        self.outages: dict[int, list[Outage]] = defaultdict(list)
        self.exports: list[Export] = []
        self.novel_errors: list[NovelError] = []
        self.activations: dict[int, Activation] = {}
        self.dedicated: set[int] = set()  # orgs that take part in one scenario and no other
        self.surges: list[Surge] = []
        self.error_spans: dict[int, list[ErrorSpan]] = defaultdict(list)
        self.rate_ramps: dict[int, list[Ramp]] = defaultdict(list)
        self.latency_ramps: dict[int, list[Ramp]] = defaultdict(list)
        self.echoes: list[_Echo] = []
        self.cascades: list[Cascade] = []
        self.natural_deploys: list[int] = []
        self.fix_deploys: list[int] = []
        self.deploys_used: set[int] = set()  # natural deploys a scenario has claimed
        self.renames: list[RenameAt] = []
        self.injected: list[_Pending] = []
        self.quiet: list[tuple[str, int | None, int, int, int]] = []  # natural (kind, key, org, start, end)
        self.occupied: dict[int, list[tuple[int, int]]] = defaultdict(list)

    # -- natural episodes ------------------------------------------------

    def _natural_bursts(self) -> None:
        days = (self.end - self.start) / DAY
        for key in self.world.keys:
            if key.archetype != "bursty":
                continue
            draw = rng(self.seed, Stream.BURSTS, key.index)
            count = int(draw.poisson(self.cfg.archetypes["bursty"].bursts_per_day * days))
            for begin in np.sort(draw.integers(self.start, self.end, count)).tolist():
                duration, rate = self._draw_burst(draw)
                burst = Burst(key.index, begin, begin + duration, rate)
                self.natural_bursts.append(burst)
                self.bursts[key.index].append(burst)

    def _draw_burst(self, draw: np.random.Generator) -> tuple[int, float]:
        b = self.cfg.bursts
        seconds = min(draw.lognormal(math.log(b.duration_seconds_median), b.duration_log_sigma), b.duration_seconds_max)
        rate = draw.lognormal(math.log(b.rate_per_second_median), b.rate_log_sigma)
        return int(seconds * NS), float(rate)

    def _maintenance_windows(self) -> None:
        for org in self.world.orgs:
            maintenance = org.provider.maintenance
            if maintenance is None or org.maintenance_minute is None:
                continue
            tz = zone(maintenance.timezone)
            draw = rng(self.seed, Stream.MAINTENANCE, org.index)
            previous = None
            for day in _local_dates(tz, self.start, self.end):
                nominal = _wall(day, org.maintenance_minute * 60, tz)
                jitter = draw.uniform(-maintenance.jitter_minutes, maintenance.jitter_minutes)
                minutes = draw.uniform(maintenance.duration_minutes.low, maintenance.duration_minutes.high)
                begin = nominal + int(jitter * MINUTE)
                end = begin + int(minutes * MINUTE)
                # A provider that schedules in local time moves in UTC when its
                # zone changes offset; recorded so a scorer can isolate those days.
                shift = 0 if previous is None else (nominal - previous - DAY) // MINUTE
                previous = nominal
                if end > self.start and begin < self.end:
                    detail = {"provider": org.provider.name, "utc_shift_minutes": shift}
                    self.windows.append(_Window(org.index, begin, end, detail))

    def _exports(self) -> None:
        ex = self.cfg.exports
        days = (self.end - self.start) / DAY
        for key in self.world.keys:
            draw = rng(self.seed, Stream.EXPORTS, key.index)
            count = int(draw.poisson(ex.ad_hoc_per_day * days))
            for begin in np.sort(draw.integers(self.start, self.end, count)).tolist():
                size = draw.lognormal(math.log(ex.ad_hoc_bytes_median), ex.ad_hoc_bytes_log_sigma)
                self.ad_hoc_exports.append(self._export(draw, key, begin, size))
            weekly = key.weekly_export
            if weekly is None:
                continue
            tz = zone(self.world.orgs[key.org].timezone)
            for day in _local_dates(tz, self.start, self.end):
                if day.weekday() != weekly.weekday:
                    continue
                jitter = draw.uniform(-ex.scheduled_jitter_minutes, ex.scheduled_jitter_minutes)
                begin = _wall(day, weekly.minute_of_day * 60, tz) + int(jitter * MINUTE)
                size = draw.lognormal(math.log(ex.scheduled_bytes_median), ex.scheduled_bytes_log_sigma)
                export = self._export(draw, key, begin, size)
                if export.parts[0][0] < self.end and export.parts[-1][0] >= self.start:
                    self.weekly_exports.append(export)

    def _export(self, draw: np.random.Generator, key: Key, begin: int, size: float) -> Export:
        ex = self.cfg.exports
        total = max(1, int(size))
        count = max(1, math.ceil(total / ex.part_bytes))
        gaps = draw.uniform(ex.part_interval_seconds.low, ex.part_interval_seconds.high, count - 1)
        times = [begin] + [begin + int(s * NS) for s in np.cumsum(gaps).tolist()]
        sizes = [ex.part_bytes] * (count - 1) + [total - ex.part_bytes * (count - 1)]
        export_id = f"exp_{int(draw.integers(1 << 48)):012x}"
        return Export(key.index, export_id, tuple(zip(times, sizes, strict=True)))

    def _batches(self) -> None:
        b = self.cfg.batches
        for key in self.world.keys:
            if key.batch_minute is None:
                continue
            tz = zone(self.world.orgs[key.org].timezone)
            # The world drew batch_minute from (BATCHES, key); the trailing 0
            # keeps these draws on a stream of their own.
            draw = rng(self.seed, Stream.BATCHES, key.index, 0)
            for day in _local_dates(tz, self.start, self.end):
                jitter = draw.uniform(-b.jitter_minutes, b.jitter_minutes)
                minutes = draw.uniform(b.duration_minutes.low, b.duration_minutes.high)
                multiplier = float(draw.uniform(b.multiplier.low, b.multiplier.high))
                begin = _wall(day, key.batch_minute * 60, tz) + int(jitter * MINUTE)
                end = begin + int(minutes * MINUTE)
                if end > self.start and begin < self.end:
                    self.surges.append(Surge(key.index, begin, end, multiplier))

    def _retry_keys(self) -> list[Key]:
        return [k for k in self.world.keys if k.rate * self.scale >= self.cfg.retries.min_rate_per_second]

    def _echoes(self) -> None:
        r = self.cfg.retries
        days = (self.end - self.start) / DAY
        for key in self._retry_keys():
            draw = rng(self.seed, Stream.RETRIES, key.index)
            count = int(draw.poisson(r.echoes_per_key_per_day * days))
            for begin in np.sort(draw.integers(self.start, self.end, count)).tolist():
                blip = int(draw.uniform(r.blip_seconds.low, r.blip_seconds.high))
                branching = float(draw.uniform(r.echo_branching.low, r.echo_branching.high))
                attempts = self._cascade(draw, key, begin, blip, branching, r.echo_delay_seconds, storm=False)
                self.echoes.append(_Echo(key.index, begin, begin + blip * NS, attempts))

    def _cascade(
        self,
        draw: np.random.Generator,
        key: Key,
        begin: int,
        overload: int,
        branching: float,
        delay: float,
        storm: bool,
    ) -> tuple[int, ...]:
        """Retries per second from `begin`, as a discrete self-exciting process.

        Every failed attempt spawns `branching` retries on average, spread over
        the following seconds with an exponential delay of mean `delay`.
        Attempts fail while the dependency is overloaded, the first `overload`
        seconds, and succeed after, spawning nothing. The admission limit caps
        attempts per second. A storm's clients stop when the overload ends
        (the circuit breaker opens); an echo's retries trail off on their own.
        """
        cap = self.cfg.retries.storm_max_rate_per_second * self.scale
        decay = math.exp(-1.0 / delay)
        gain = branching * (1.0 - decay)
        first = begin // NS
        tz = zone(self.world.orgs[key.org].timezone)
        # The key's own requests failing during the overload seed the cascade.
        # These are draws from the process the generator emits, not the same
        # draws; nothing outside the plan can tell the two apart.
        originals = key.rate * self.scale * profile(np.arange(first, first + overload), tz, self.cfg.traffic)
        intensity = 0.0
        attempts: list[int] = []
        for second in range(overload if storm else overload + _ECHO_MAX_SECONDS):
            sent = int(draw.poisson(min(intensity, cap)))
            attempts.append(sent)
            failed = sent + int(draw.poisson(originals[second])) if second < overload else 0
            intensity = intensity * decay + gain * failed
            if not storm and second >= overload and intensity < _ECHO_DONE_INTENSITY:
                break
        else:
            if not storm:
                raise PlanError(f"a retry echo on {key.id} was still running after {_ECHO_MAX_SECONDS} s")
        while attempts and attempts[-1] == 0:
            attempts.pop()
        return tuple(attempts)

    def _deploys(self) -> None:
        d = self.cfg.deploys
        tz = zone(d.timezone)
        draw = rng(self.seed, Stream.DEPLOYS)
        low, high = int(d.local_hour.low * 3600), max(int(d.local_hour.high * 3600), int(d.local_hour.low * 3600) + 1)
        for day in _local_dates(tz, self.start, self.end):
            count = int(draw.poisson(d.per_weekday)) if day.weekday() < 5 else 0
            for second in sorted(set(draw.integers(low, high, count).tolist())):
                at = _wall(day, second, tz)
                if self.start <= at < self.end:
                    self.natural_deploys.append(at)

    # -- scenarios -------------------------------------------------------

    def _eligible_keys(self, predicate: Callable[[Key, Org], bool]) -> list[Key]:
        return [k for k in self.world.keys if k.org not in self.dedicated and predicate(k, self.world.orgs[k.org])]

    def _eligible_orgs(self) -> list[Org]:
        return [o for o in self.world.orgs if o.index not in self.dedicated]

    def _free(self, org: int, start: int, end: int) -> bool:
        return all(end + self.gap <= s or e + self.gap <= start for s, e in self.occupied[org])

    def _index_quiet(self) -> None:
        """Natural quiet cases, for _clashes. Includes ones a later filter drops
        (before an org arrives, during a silence); avoiding those too only
        narrows placement."""
        for b in self.natural_bursts:
            self.quiet.append(("natural_burst", b.key, self.world.keys[b.key].org, b.start, b.end))
        for w in self.windows:
            self.quiet.append(("maintenance_window", None, w.org, w.start, w.end))
        for e in self.weekly_exports:
            self.quiet.append(("scheduled_export", e.key, self.world.keys[e.key].org, e.parts[0][0], e.parts[-1][0]))
        for s in self.surges:
            self.quiet.append(("batch_surge", s.key, self.world.keys[s.key].org, s.start, s.end))
        for e in self.echoes:
            self.quiet.append(("retry_echo", e.key, self.world.keys[e.key].org, e.start, e.end))

    def _clashes(self, kind: str, orgs: tuple[int, ...], key: int | None, start: int, end: int) -> bool:
        """Would a page on this alert case also land on a natural quiet case,
        by the scorer's rules, even with the right signal? A case on whole orgs
        shares pages with everything in them. A key-level case shares them with
        its key's cases; it would also share them with its org's org-level
        ones, but the only natural one, a maintenance window, shows `health`,
        which no key-level kind does."""
        signals = SIGNALS[kind]
        low, high = start - _QUIET_MARGIN, end + _QUIET_MARGIN
        for quiet, k, org, s, e in self.quiet:
            if e <= low or s >= high or not SIGNALS[quiet] & signals:
                continue
            # An overrun replaces the window it starts with; that one is not a neighbour.
            if kind == "maintenance_overrun" and quiet == "maintenance_window" and s == start:
                continue
            if key is not None and k == key:
                return True
            if key is None and org in orgs:
                return True
        return False

    def _place(
        self, kind: str, count: int, candidates: int, sample: Callable[[], _Candidate | None], alert: bool = True
    ) -> None:
        if count and not candidates:
            raise PlanError(f"scenarios.{kind}: no eligible targets in this world; see the README for each kind's rule")
        placed = 0
        for _ in range(_ATTEMPTS_PER_SCENARIO * count):
            if placed == count:
                return
            candidate = sample()
            if candidate is None:
                continue
            orgs, key, start, end, commit = candidate
            if start < self.scored_from or end > self.end or not all(self._free(o, start, end) for o in orgs):
                continue
            if alert and self._clashes(kind, orgs, key, start, end):
                continue
            for org in orgs:
                self.occupied[org].append((start, end))
            commit()
            placed += 1
        if placed < count:
            raise PlanError(
                f"scenarios.{kind}: placed {placed} of {count}; lower the count or "
                "scenarios.min_gap_hours, lengthen run.scored_days, or add orgs"
            )

    def _local_time(
        self, draw: np.random.Generator, tz: ZoneInfo, hours: Range, weekdays_only: bool = False
    ) -> int | None:
        day = datetime.fromtimestamp(
            (self.scored_from + int(draw.integers(self.cfg.scored_days)) * DAY + 12 * HOUR) // NS, tz
        ).date()
        second = int(draw.integers(int(hours.low * 3600), max(int(hours.high * 3600), int(hours.low * 3600) + 1)))
        if weekdays_only and day.weekday() >= 5:
            return None
        return _wall(day, second, tz)

    def _scenarios(self) -> None:
        sc = self.cfg.scenarios
        draw = rng(self.seed, Stream.SCENARIOS)

        # Onboarding first: an org that only appears mid-run is excluded from
        # every other scenario, so its cold start is the only thing it tests.
        ramp = int(sc.onboarding_ramp_days * DAY)
        newcomers = [o for o in self.world.orgs if not o.fixed]

        def onboarding() -> _Candidate | None:
            org = newcomers[int(draw.integers(len(newcomers)))]
            if org.index in self.activations:
                return None
            begin = int(draw.integers(self.scored_from, max(self.scored_from + 1, self.end - ramp)))

            def commit() -> None:
                self.activations[org.index] = Activation(begin, begin + ramp)
                self.dedicated.add(org.index)
                self.injected.append(
                    _Pending("onboarding", "onboarding", "quiet", org.index, None, begin, begin + ramp)
                )

            return (org.index,), None, begin, begin + ramp, commit

        self._place("onboarding", sc.onboarding, len(newcomers), onboarding, alert=False)
        self._organic_growth()

        def mixed_org(k: Key, o: Org) -> bool:
            return any(self.world.keys[j].archetype == "bursty" for j in o.keys)

        steady = self._eligible_keys(lambda k, o: k.archetype == "steady" and mixed_org(k, o))
        burst_keys: set[int] = set()

        def burst_on_steady_key() -> _Candidate | None:
            key = _pick(draw, steady, burst_keys, lambda k: k.index)
            duration, rate = self._draw_burst(draw)
            begin = int(draw.integers(self.scored_from, self.end - duration))
            detail = {"rate_per_second": rate * self.scale, "baseline_rate_per_second": key.rate * self.scale}

            def commit() -> None:
                burst_keys.add(key.index)
                self.bursts[key.index].append(Burst(key.index, begin, begin + duration, rate))
                self.injected.append(
                    _Pending(
                        "burst_on_steady_key", "burst", "alert", key.org, key.index, begin, begin + duration, detail
                    )
                )

            return (key.org,), key.index, begin, begin + duration, commit

        self._place("burst_on_steady_key", sc.burst_on_steady_key, len(steady), burst_on_steady_key)

        unmaintained = [o for o in self._eligible_orgs() if o.provider.maintenance is None]
        outage_orgs: set[int] = set()

        def unplanned_outage() -> _Candidate | None:
            org = _pick(draw, unmaintained, outage_orgs, lambda o: o.index)
            begin = self._local_time(draw, zone(org.timezone), sc.outage_local_hour)
            minutes = draw.uniform(sc.outage_minutes.low, sc.outage_minutes.high)
            if begin is None:
                return None
            end = begin + int(minutes * MINUTE)

            def commit() -> None:
                outage_orgs.add(org.index)
                self.outages[org.index].append(Outage(org.index, begin, end))
                detail = {"provider": org.provider.name}
                self.injected.append(
                    _Pending("unplanned_outage", "outage", "alert", org.index, None, begin, end, detail)
                )

            return (org.index,), None, begin, end, commit

        self._place("unplanned_outage", sc.unplanned_outage, len(unmaintained), unplanned_outage)

        windows = [
            w for w in self.windows if w.start >= self.scored_from and w.end <= self.end and w.org not in self.dedicated
        ]
        overrun_orgs: set[int] = set()

        def maintenance_overrun() -> _Candidate | None:
            window = _pick(draw, windows, overrun_orgs, lambda w: w.org)
            minutes = draw.uniform(sc.overrun_minutes.low, sc.overrun_minutes.high)
            if window.overrun:
                return None
            end = window.start + int(minutes * MINUTE)

            def commit() -> None:
                overrun_orgs.add(window.org)
                window.overrun = True
                detail = {**window.detail, "scheduled_minutes": round((window.end - window.start) / MINUTE, 2)}
                self.injected.append(
                    _Pending("maintenance_overrun", "outage", "alert", window.org, None, window.start, end, detail)
                )
                self.outages[window.org].append(Outage(window.org, window.start, end))

            return (window.org,), None, window.start, end, commit

        self._place("maintenance_overrun", sc.maintenance_overrun, len(windows), maintenance_overrun)

        anyone = self._eligible_keys(lambda k, o: True)
        templates = [self.cfg.novel_errors[i] for i in draw.permutation(len(self.cfg.novel_errors)).tolist()]
        window_ns = int(sc.novel_error_window_minutes * MINUTE)
        novel_keys: set[int] = set()

        def novel_error() -> _Candidate | None:
            key = _pick(draw, anyone, novel_keys, lambda k: k.index)
            begin = int(draw.integers(self.scored_from, self.end - window_ns))
            times = np.sort(draw.integers(begin, begin + window_ns + 1, sc.novel_error_count)).tolist()

            def commit() -> None:
                novel_keys.add(key.index)
                template = templates.pop(0)
                self.novel_errors.extend(NovelError(key.index, t, template) for t in times)
                detail = {"template": template.text, "count": sc.novel_error_count}
                self.injected.append(
                    _Pending("novel_error", "error", "alert", key.org, key.index, begin, begin + window_ns, detail)
                )

            return (key.org,), key.index, begin, begin + window_ns, commit

        self._place("novel_error", sc.novel_error, len(anyone), novel_error)

        unscheduled = self._eligible_keys(lambda k, o: k.weekly_export is None)
        ex = self.cfg.exports
        export_keys: set[int] = set()

        def unscheduled_bulk_export() -> _Candidate | None:
            key = _pick(draw, unscheduled, export_keys, lambda k: k.index)
            begin = self._local_time(draw, zone(self.world.orgs[key.org].timezone), ex.scheduled_local_hour)
            size = draw.lognormal(math.log(ex.scheduled_bytes_median), ex.scheduled_bytes_log_sigma)
            if begin is None:
                return None
            export = self._export(draw, key, begin, size)
            end = export.parts[-1][0] + NS

            def commit() -> None:
                export_keys.add(key.index)
                self.exports.append(export)
                detail = {"bytes": sum(b for _, b in export.parts)}
                self.injected.append(
                    _Pending("unscheduled_bulk_export", "egress", "alert", key.org, key.index, begin, end, detail)
                )

            return (key.org,), key.index, begin, end, commit

        self._place("unscheduled_bulk_export", sc.unscheduled_bulk_export, len(unscheduled), unscheduled_bulk_export)

        loud = self._eligible_keys(
            lambda k, o: k.archetype != "bursty" and k.rate * self.scale >= sc.silence_min_rate_per_second
        )
        silence_ns = int(sc.silence_hours * HOUR)
        silent_keys: set[int] = set()

        def key_silence() -> _Candidate | None:
            key = _pick(draw, loud, silent_keys, lambda k: k.index)
            begin = self._local_time(
                draw, zone(self.world.orgs[key.org].timezone), sc.silence_local_hour, weekdays_only=True
            )
            if begin is None:
                return None

            def commit() -> None:
                silent_keys.add(key.index)
                self.silences[key.index].append(Silence(key.index, begin, begin + silence_ns))
                detail = {"baseline_rate_per_second": key.rate * self.scale}
                self.injected.append(
                    _Pending("key_silence", "silence", "alert", key.org, key.index, begin, begin + silence_ns, detail)
                )

            return (key.org,), key.index, begin, begin + silence_ns, commit

        self._place("key_silence", sc.key_silence, len(loud), key_silence)

        self._retry_storms()
        self._error_rate_shifts()
        self._provider_outages()
        self._bad_deploys()
        self._renamed_errors()
        self._latency_leaks()

    # Each kind below draws from a stream of its own, so adding or resizing one
    # leaves the others' draws alone. Placement still depends on order: a kind
    # avoids the slots earlier kinds took.

    def _organic_growth(self) -> None:
        sc = self.cfg.scenarios
        draw = rng(self.seed, Stream.SCENARIOS, 1)
        orgs = self._eligible_orgs()

        def organic_growth() -> _Candidate | None:
            org = orgs[int(draw.integers(len(orgs)))]
            days = draw.uniform(sc.growth_days.low, sc.growth_days.high)
            factor = float(draw.uniform(sc.growth_factor.low, sc.growth_factor.high))
            if org.index in self.dedicated:
                return None
            begin = int(draw.integers(self.scored_from, self.end))
            end = begin + int(days * DAY)

            def commit() -> None:
                self.dedicated.add(org.index)
                self.rate_ramps[org.index].append(Ramp(org.index, begin, end, factor, holds=True))
                detail = {"factor": factor}
                self.injected.append(_Pending("organic_growth", "drift", "quiet", org.index, None, begin, end, detail))

            # The org's rate stays raised to the end of the run; being dedicated
            # keeps every later scenario out of it.
            return (org.index,), None, begin, end, commit

        self._place("organic_growth", sc.organic_growth, len(orgs), organic_growth, alert=False)

    def _retry_storms(self) -> None:
        sc, r = self.cfg.scenarios, self.cfg.retries
        draw = rng(self.seed, Stream.SCENARIOS, 2)
        keys = [k for k in self._retry_keys() if k.org not in self.dedicated]
        cap = r.storm_max_rate_per_second * self.scale
        used: set[int] = set()

        def retry_storm() -> _Candidate | None:
            key = _pick(draw, keys, used, lambda k: k.index)
            minutes = draw.uniform(r.storm_minutes.low, r.storm_minutes.high)
            branching = float(draw.uniform(r.storm_branching.low, r.storm_branching.high))
            blip = int(draw.uniform(r.blip_seconds.low, r.blip_seconds.high))
            begin = int(draw.integers(self.scored_from, self.end))
            overload = max(int(minutes * 60), blip)
            end = begin + overload * NS
            if end > self.end:
                return None
            attempts = self._cascade(draw, key, begin, overload, branching, r.storm_delay_seconds, storm=True)
            # A supercritical cascade can still die out in its first seconds;
            # that run is an echo, not a storm, so draw again.
            if max(attempts, default=0) < cap:
                return None

            def commit() -> None:
                used.add(key.index)
                self.cascades.append(Cascade(key.index, begin, end, attempts))
                self.error_spans[key.index].append(ErrorSpan(key.index, begin, end, 1.0, OVERLOADED))
                detail = {"branching": branching, "retries": sum(attempts), "blip_seconds": blip}
                self.injected.append(_Pending("retry_storm", "retry", "alert", key.org, key.index, begin, end, detail))

            return (key.org,), key.index, begin, end, commit

        self._place("retry_storm", sc.retry_storm, len(keys), retry_storm)

    def _error_rate_shifts(self) -> None:
        sc, b = self.cfg.scenarios, self.cfg.batches
        draw = rng(self.seed, Stream.SCENARIOS, 3)
        keys = self._eligible_keys(lambda k, o: k.rate * self.scale >= sc.error_shift_min_rate_per_second)
        used: set[int] = set()

        def error_rate_shift() -> _Candidate | None:
            # Same duration and multiplier as a daily batch: a batch multiplies
            # the volume and the errors with it, a shift only the errors.
            key = _pick(draw, keys, used, lambda k: k.index)
            minutes = draw.uniform(b.duration_minutes.low, b.duration_minutes.high)
            multiplier = float(draw.uniform(b.multiplier.low, b.multiplier.high))
            begin = int(draw.integers(self.scored_from, self.end))
            end = begin + int(minutes * MINUTE)
            share = min(1.0, self.cfg.traffic.server_error_share * multiplier)

            def commit() -> None:
                used.add(key.index)
                self.error_spans[key.index].append(ErrorSpan(key.index, begin, end, share, SERVER_ERROR))
                detail = {"multiplier": multiplier, "server_error_share": share}
                self.injected.append(
                    _Pending("error_rate_shift", "error_rate", "alert", key.org, key.index, begin, end, detail)
                )

            return (key.org,), key.index, begin, end, commit

        self._place("error_rate_shift", sc.error_rate_shift, len(keys), error_rate_shift)

    def _provider_members(self) -> dict[str, list[Org]]:
        members: dict[str, list[Org]] = defaultdict(list)
        for org in self._eligible_orgs():
            members[org.provider.name].append(org)
        return members

    def _outage_providers(self) -> list[tuple[str, list[Org]]]:
        # Dedicated orgs stay out of it even on the failing provider, so each
        # still tests only the one thing it was set aside for.
        members = self._provider_members()
        return [(p, members[p]) for p in sorted(members) if len(members[p]) >= _PROVIDER_OUTAGE_MIN_ORGS]

    def _provider_outages(self) -> None:
        sc = self.cfg.scenarios
        draw = rng(self.seed, Stream.SCENARIOS, 4)
        providers = self._outage_providers()
        used: set[int] = set()

        def provider_outage() -> _Candidate | None:
            index = _pick(draw, list(range(len(providers))), used, lambda i: i)
            provider, orgs = providers[index]
            minutes = draw.uniform(sc.provider_outage_minutes.low, sc.provider_outage_minutes.high)
            begin = int(draw.integers(self.scored_from, self.end))
            end = begin + int(minutes * MINUTE)

            def commit() -> None:
                used.add(index)
                for org in orgs:
                    self.outages[org.index].append(Outage(org.index, begin, end))
                detail = {"provider": provider, "orgs": sorted(o.id for o in orgs)}
                self.injected.append(_Pending("provider_outage", "outage", "alert", None, None, begin, end, detail))

            return tuple(o.index for o in orgs), None, begin, end, commit

        self._place("provider_outage", sc.provider_outage, len(providers), provider_outage)

    def _scored_deploys(self) -> list[int]:
        return [t for t in self.natural_deploys if t >= self.scored_from]

    def _bad_deploys(self) -> None:
        sc = self.cfg.scenarios
        draw = rng(self.seed, Stream.SCENARIOS, 5)
        deploys = self._scored_deploys()
        members = self._provider_members()
        providers = [(p, members[p]) for p in sorted(members)]

        def bad_deploy() -> _Candidate | None:
            at = deploys[int(draw.integers(len(deploys)))]
            provider, orgs = providers[int(draw.integers(len(providers)))]
            minutes = draw.uniform(sc.bad_deploy_minutes.low, sc.bad_deploy_minutes.high)
            share = float(draw.uniform(sc.bad_deploy_error_share.low, sc.bad_deploy_error_share.high))
            if at in self.deploys_used:
                return None
            fix = at + int(minutes * MINUTE)

            def commit() -> None:
                # The fix ships as one more forward deploy. A rollback marker
                # would name the bad release after the fact, and a detector
                # paging on it would score a detection it never made.
                self.deploys_used.add(at)
                self.fix_deploys.append(fix)
                for org in orgs:
                    for key in org.keys:
                        self.error_spans[key].append(ErrorSpan(key, at, fix, share, SERVER_ERROR))
                detail = {"provider": provider, "orgs": sorted(o.id for o in orgs), "server_error_share": share}
                self.injected.append(_Pending("bad_deploy", "deploy", "alert", None, None, at, fix, detail))

            return tuple(o.index for o in orgs), None, at, fix, commit

        self._place("bad_deploy", sc.bad_deploy, len(deploys) and len(providers), bad_deploy)

    def _renamed_errors(self) -> None:
        sc = self.cfg.scenarios
        draw = rng(self.seed, Stream.SCENARIOS, 6)
        deploys = self._scored_deploys()
        renames = [self.cfg.renames[i] for i in draw.permutation(len(self.cfg.renames)).tolist()]

        def renamed_error() -> _Candidate | None:
            at = deploys[int(draw.integers(len(deploys)))]
            if at in self.deploys_used:
                return None

            def commit() -> None:
                self.deploys_used.add(at)
                rename = renames.pop(0)
                self.renames.append(RenameAt(at, rename.template, rename.to))

            # Fleet-wide and harmless: it occupies no org. Its truth comes from
            # the emitted lines, in logsim.labels.
            return (), None, at, at + 1, commit

        self._place("renamed_error", sc.renamed_error, len(deploys), renamed_error, alert=False)

    def _latency_leaks(self) -> None:
        sc = self.cfg.scenarios
        draw = rng(self.seed, Stream.SCENARIOS, 7)
        orgs = self._eligible_orgs()
        used: set[int] = set()

        def latency_leak() -> _Candidate | None:
            org = _pick(draw, orgs, used, lambda o: o.index)
            hours = draw.uniform(sc.leak_hours.low, sc.leak_hours.high)
            factor = float(draw.uniform(sc.leak_factor.low, sc.leak_factor.high))
            begin = int(draw.integers(self.scored_from, self.end))
            end = begin + int(hours * HOUR)

            def commit() -> None:
                used.add(org.index)
                self.latency_ramps[org.index].append(Ramp(org.index, begin, end, factor, holds=False))
                detail = {"factor": factor}
                self.injected.append(_Pending("latency_leak", "drift", "alert", org.index, None, begin, end, detail))

            return (org.index,), None, begin, end, commit

        self._place("latency_leak", sc.latency_leak, len(orgs), latency_leak)

    # -- assembly --------------------------------------------------------

    def build(self) -> Plan:
        self._natural_bursts()
        self._maintenance_windows()
        self._exports()
        self._batches()
        self._echoes()
        self._deploys()
        self._index_quiet()
        self._scenarios()

        active = self._is_active
        pending = list(self.injected)
        for burst in self.natural_bursts:
            org = self.world.keys[burst.key].org
            if burst.start < self.end and burst.end > self.start and active(org, burst.start):
                detail = {"rate_per_second": burst.rate * self.scale}
                pending.append(
                    _Pending("natural_burst", "burst", "quiet", org, burst.key, burst.start, burst.end, detail)
                )
        for window in self.windows:
            if window.overrun or not active(window.org, window.start):
                continue
            self.outages[window.org].append(Outage(window.org, window.start, window.end))
            pending.append(
                _Pending(
                    "maintenance_window", "outage", "quiet", window.org, None, window.start, window.end, window.detail
                )
            )
        for export in self.weekly_exports:
            org = self.world.keys[export.key].org
            if not active(org, export.parts[0][0]):
                continue
            self.exports.append(export)
            begin, end = export.parts[0][0], export.parts[-1][0] + NS
            detail = {"bytes": sum(b for _, b in export.parts)}
            pending.append(_Pending("scheduled_export", "egress", "quiet", org, export.key, begin, end, detail))
        # Same reasoning as background errors: a silenced key exports nothing.
        self.exports.extend(
            e
            for e in self.ad_hoc_exports
            if active(self.world.keys[e.key].org, e.parts[0][0])
            and not any(s.start <= e.parts[-1][0] and e.parts[0][0] < s.end for s in self.silences.get(e.key, ()))
        )
        surges: dict[int, list[Surge]] = defaultdict(list)
        for surge in self.surges:
            org = self.world.keys[surge.key].org
            if not active(org, surge.start):
                continue
            surges[surge.key].append(surge)
            detail = {"multiplier": surge.multiplier}
            pending.append(
                _Pending("batch_surge", "error_rate", "quiet", org, surge.key, surge.start, surge.end, detail)
            )
        for echo in self.echoes:
            org = self.world.keys[echo.key].org
            # No echo overlaps a silence: _clashes keeps key_silence clear of them.
            if not active(org, echo.start):
                continue
            self.cascades.append(Cascade(echo.key, echo.start, echo.overload_end, echo.attempts))
            self.error_spans[echo.key].append(ErrorSpan(echo.key, echo.start, echo.overload_end, 1.0, OVERLOADED))
            detail = {"retries": sum(echo.attempts)}
            pending.append(_Pending("retry_echo", "retry", "quiet", org, echo.key, echo.start, echo.end, detail))

        times = sorted(self.natural_deploys + self.fix_deploys)
        if len(set(times)) != len(times):
            raise PlanError("two deploys landed on the same nanosecond; choose another seed")
        deploys = tuple(Deploy(t, f"r{i:04d}") for i, t in enumerate(times, start=1))
        release = {d.time: d.release for d in deploys}
        for p in pending:
            if p.kind == "bad_deploy":
                p.detail = {"release": release[p.start], "fix_release": release[p.end], **p.detail}

        pending.sort(key=lambda p: (p.start, p.kind, -1 if p.org is None else p.org, -1 if p.key is None else p.key))
        truth = tuple(
            Truth(
                id=f"t{i:05d}",
                kind=p.kind,
                pair=p.pair,
                expect=p.expect,
                org=p.org,
                key=p.key,
                start=p.start,
                end=p.end,
                scored=p.start >= self.scored_from,
                detail=p.detail,
            )
            for i, p in enumerate(pending, start=1)
        )
        return Plan(
            start=self.start,
            scored_from=self.scored_from,
            end=self.end,
            bursts={k: tuple(v) for k, v in self.bursts.items()},
            silences={k: tuple(v) for k, v in self.silences.items()},
            outages={k: tuple(sorted(v, key=lambda o: o.start)) for k, v in self.outages.items()},
            exports=tuple(sorted(self.exports, key=lambda e: (e.parts[0][0], e.id))),
            novel_errors=tuple(sorted(self.novel_errors, key=lambda n: n.time)),
            activations=dict(self.activations),
            surges={k: tuple(sorted(v, key=lambda s: s.start)) for k, v in surges.items()},
            error_spans={k: tuple(sorted(v, key=lambda s: s.start)) for k, v in self.error_spans.items()},
            rate_ramps={k: tuple(v) for k, v in self.rate_ramps.items()},
            latency_ramps={k: tuple(v) for k, v in self.latency_ramps.items()},
            cascades=tuple(sorted(self.cascades, key=lambda c: (c.start, c.key))),
            deploys=deploys,
            renames=tuple(sorted(self.renames, key=lambda r: r.time)),
            truth=truth,
        )

    def _is_active(self, org: int, at: int) -> bool:
        return _active(self.activations, org, at)


def _active(activations: dict[int, Activation], org: int, at: int) -> bool:
    activation = activations.get(org)
    return activation is None or at >= activation.start


_Candidate = tuple[tuple[int, ...], int | None, int, int, Callable[[], None]]  # (orgs, key, start, end, commit)


def _pick[T](draw: np.random.Generator, items: list[T], used: set[int], ident: Callable[[T], int]) -> T:
    """A target this scenario kind has not used yet, while any remain. A detector
    that learned from an earlier injection on the same entity would otherwise be
    graded on its own adaptation rather than on the case in front of it."""
    fresh = [item for item in items if ident(item) not in used] or items
    return fresh[int(draw.integers(len(fresh)))]


def _local_dates(tz: ZoneInfo, start: int, end: int) -> Iterator[date]:
    """Every local date whose events could touch [start, end), with a day of slack."""
    day = datetime.fromtimestamp(start // NS, tz).date() - timedelta(days=1)
    last = datetime.fromtimestamp(end // NS, tz).date()
    while day <= last:
        yield day
        day += timedelta(days=1)


def _wall(day: date, second_of_day: int, tz: ZoneInfo) -> int:
    """UTC nanoseconds for a local wall-clock time. Ambiguous and skipped times
    resolve with fold=0, as zoneinfo defines; the result is the same everywhere."""
    moment = datetime.combine(day, time(), tzinfo=tz) + timedelta(seconds=second_of_day)
    return int(moment.timestamp()) * NS
