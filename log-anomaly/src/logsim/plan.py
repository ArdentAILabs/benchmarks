"""Everything a run does beyond steady traffic, decided before any record exists.

The plan holds the natural episodes (bursts, maintenance windows, exports), the
injected scenarios, and the ground truth labelling both. The record generator
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
from logsim.repro import Stream, rng, zone
from logsim.world import Key, Org, World

NS = 10**9
MINUTE = 60 * NS
HOUR = 60 * MINUTE
DAY = 24 * HOUR

# Draws per scenario before the schedule is declared unplaceable.
_ATTEMPTS_PER_SCENARIO = 1000


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
class Activation:
    start: int  # no records for the org before this
    ramp_end: int  # traffic grows linearly from zero until this


@dataclass(frozen=True)
class Truth:
    id: str
    kind: str
    pair: str  # matched-pair family the case belongs to
    expect: str  # "alert" or "quiet"
    org: int
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
class _Pending:
    kind: str
    pair: str
    expect: str
    org: int
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
        self.injected: list[_Pending] = []
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

    # -- scenarios -------------------------------------------------------

    def _onboarding_orgs(self) -> set[int]:
        return set(self.activations)

    def _eligible_keys(self, predicate: Callable[[Key, Org], bool]) -> list[Key]:
        onboarding = self._onboarding_orgs()
        return [k for k in self.world.keys if k.org not in onboarding and predicate(k, self.world.orgs[k.org])]

    def _free(self, org: int, start: int, end: int) -> bool:
        return all(end + self.gap <= s or e + self.gap <= start for s, e in self.occupied[org])

    def _place(self, kind: str, count: int, candidates: int, sample: Callable[[], _Candidate | None]) -> None:
        if count and not candidates:
            raise PlanError(f"scenarios.{kind}: no eligible targets in this world; see the README for each kind's rule")
        placed = 0
        for _ in range(_ATTEMPTS_PER_SCENARIO * count):
            if placed == count:
                return
            candidate = sample()
            if candidate is None:
                continue
            org, start, end, commit = candidate
            if start < self.scored_from or end > self.end or not self._free(org, start, end):
                continue
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
                self.injected.append(
                    _Pending("onboarding", "onboarding", "quiet", org.index, None, begin, begin + ramp)
                )

            return org.index, begin, begin + ramp, commit

        self._place("onboarding", sc.onboarding, len(newcomers), onboarding)

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

            return key.org, begin, begin + duration, commit

        self._place("burst_on_steady_key", sc.burst_on_steady_key, len(steady), burst_on_steady_key)

        onboarding_orgs = self._onboarding_orgs()
        unmaintained = [o for o in self.world.orgs if o.provider.maintenance is None and o.index not in onboarding_orgs]
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

            return org.index, begin, end, commit

        self._place("unplanned_outage", sc.unplanned_outage, len(unmaintained), unplanned_outage)

        windows = [
            w
            for w in self.windows
            if w.start >= self.scored_from and w.end <= self.end and w.org not in onboarding_orgs
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

            return window.org, window.start, end, commit

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

            return key.org, begin, begin + window_ns, commit

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

            return key.org, begin, end, commit

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

            return key.org, begin, begin + silence_ns, commit

        self._place("key_silence", sc.key_silence, len(loud), key_silence)

    # -- assembly --------------------------------------------------------

    def build(self) -> Plan:
        self._natural_bursts()
        self._maintenance_windows()
        self._exports()
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

        pending.sort(key=lambda p: (p.start, p.kind, p.org, -1 if p.key is None else p.key))
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
            truth=truth,
        )

    def _is_active(self, org: int, at: int) -> bool:
        return _active(self.activations, org, at)


def _active(activations: dict[int, Activation], org: int, at: int) -> bool:
    activation = activations.get(org)
    return activation is None or at >= activation.start


_Candidate = tuple[int, int, int, Callable[[], None]]  # (org, start, end, commit)


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
