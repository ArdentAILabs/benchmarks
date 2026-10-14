"""Strict loader for simulation configs.

A config describes a distribution over worlds and scenario schedules; the seed
picks one of each. Every knob is required and unknown keys are rejected, so an
omitted field or a typo fails at load time instead of quietly simulating
something other than what the file says.
"""

from __future__ import annotations

import hashlib
import string
import tomllib
from dataclasses import dataclass
from pathlib import Path

from logsim.repro import zone

ARCHETYPES = ("steady", "bursty", "overdispersed")
SEVERITIES = ("WARN", "ERROR")
TEMPLATE_FIELDS = frozenset({"ms", "id", "n"})


class ConfigError(ValueError):
    pass


@dataclass(frozen=True)
class Range:
    low: float
    high: float


@dataclass(frozen=True)
class Maintenance:
    timezone: str
    start_hour: Range  # provider-local hour an org's daily window opens in
    duration_minutes: Range
    jitter_minutes: float


@dataclass(frozen=True)
class Provider:
    name: str
    weight: float
    maintenance: Maintenance | None


@dataclass(frozen=True)
class Traffic:
    rate_scale: float  # multiplies every request rate, bursts included
    diurnal_amplitude: float
    diurnal_peak_hour: float
    weekend_factor: float


@dataclass(frozen=True)
class FixedOrg:
    keys: tuple[str, ...]


@dataclass(frozen=True)
class Orgs:
    count: int  # randomly drawn orgs, in addition to the fixed ones
    keys_per_org: tuple[int, int]
    timezones: tuple[str, ...]
    fixed: tuple[FixedOrg, ...]


@dataclass(frozen=True)
class KeyArchetype:
    weight: float
    rate_per_second: Range  # weekday mean, before rate_scale
    bursts_per_day: float  # bursty only; 0 for the others
    gamma_shape: Range | None  # overdispersed only


@dataclass(frozen=True)
class Bursts:
    rate_per_second_median: float
    rate_log_sigma: float
    duration_seconds_median: float
    duration_log_sigma: float
    duration_seconds_max: float


@dataclass(frozen=True)
class Exports:
    part_bytes: int
    part_interval_seconds: Range
    ad_hoc_per_day: float
    ad_hoc_bytes_median: float
    ad_hoc_bytes_log_sigma: float
    scheduled_share: float
    scheduled_bytes_median: float
    scheduled_bytes_log_sigma: float
    scheduled_local_hour: Range
    scheduled_jitter_minutes: float


@dataclass(frozen=True)
class Health:
    interval_seconds: int
    latency_ms_median: float
    latency_log_sigma: float


@dataclass(frozen=True)
class MessageTemplate:
    service: str
    severity: str
    text: str


@dataclass(frozen=True)
class BackgroundError:
    template: MessageTemplate
    rate_per_hour: float  # fleet-wide


@dataclass(frozen=True)
class Scenarios:
    burst_on_steady_key: int
    unplanned_outage: int
    maintenance_overrun: int
    novel_error: int
    unscheduled_bulk_export: int
    key_silence: int
    onboarding: int
    min_gap_hours: float
    outage_minutes: Range
    outage_local_hour: Range
    overrun_minutes: Range
    novel_error_count: int
    novel_error_window_minutes: float
    silence_hours: float
    silence_local_hour: Range
    silence_min_rate_per_second: float
    onboarding_ramp_days: float


@dataclass(frozen=True)
class Config:
    sha256: str
    warmup_days: int
    scored_days: int
    traffic: Traffic
    providers: tuple[Provider, ...]
    orgs: Orgs
    archetypes: dict[str, KeyArchetype]
    bursts: Bursts
    exports: Exports
    health: Health
    background_errors: tuple[BackgroundError, ...]
    novel_errors: tuple[MessageTemplate, ...]
    scenarios: Scenarios


class _Table:
    """A TOML table that tracks which keys were read, so leftovers are errors."""

    def __init__(self, data: object, path: str):
        if not isinstance(data, dict):
            raise ConfigError(f"{path}: expected a table")
        self._data = data
        self._path = path
        self._seen: set[str] = set()

    def _raw(self, key: str) -> object:
        if key not in self._data:
            raise ConfigError(f"{self._path}.{key}: required")
        self._seen.add(key)
        return self._data[key]

    def has(self, key: str) -> bool:
        return key in self._data

    def number(self, key: str, low: float | None = None, high: float | None = None) -> float:
        value = self._raw(key)
        if isinstance(value, bool) or not isinstance(value, int | float):
            raise ConfigError(f"{self._path}.{key}: expected a number")
        return _bounded(float(value), f"{self._path}.{key}", low, high)

    def integer(self, key: str, low: int | None = None) -> int:
        value = self._raw(key)
        if isinstance(value, bool) or not isinstance(value, int):
            raise ConfigError(f"{self._path}.{key}: expected an integer")
        return int(_bounded(value, f"{self._path}.{key}", low, None))

    def text(self, key: str) -> str:
        value = self._raw(key)
        if not isinstance(value, str) or not value:
            raise ConfigError(f"{self._path}.{key}: expected a non-empty string")
        return value

    def texts(self, key: str) -> tuple[str, ...]:
        value = self._raw(key)
        if not isinstance(value, list) or not value or not all(isinstance(v, str) and v for v in value):
            raise ConfigError(f"{self._path}.{key}: expected a non-empty list of strings")
        return tuple(value)

    def range(self, key: str, low: float | None = None, high: float | None = None) -> Range:
        value = self._raw(key)
        path = f"{self._path}.{key}"
        if (
            not isinstance(value, list)
            or len(value) != 2
            or any(isinstance(v, bool) or not isinstance(v, int | float) for v in value)
        ):
            raise ConfigError(f"{path}: expected [low, high]")
        lo, hi = (_bounded(float(v), path, low, high) for v in value)
        if lo > hi:
            raise ConfigError(f"{path}: low exceeds high")
        return Range(lo, hi)

    def table(self, key: str) -> _Table:
        return _Table(self._raw(key), f"{self._path}.{key}")

    def tables(self, key: str) -> list[_Table]:
        value = self._raw(key)
        if not isinstance(value, list):
            raise ConfigError(f"{self._path}.{key}: expected an array of tables")
        return [_Table(item, f"{self._path}.{key}[{i}]") for i, item in enumerate(value)]

    def close(self) -> None:
        unknown = sorted(set(self._data) - self._seen)
        if unknown:
            raise ConfigError(f"{self._path}: unknown keys {unknown}")


def _bounded(value: float, path: str, low: float | None, high: float | None) -> float:
    if low is not None and value < low:
        raise ConfigError(f"{path}: must be >= {low}")
    if high is not None and value > high:
        raise ConfigError(f"{path}: must be <= {high}")
    return value


def load(path: Path) -> Config:
    raw = path.read_bytes()
    try:
        data = tomllib.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, tomllib.TOMLDecodeError) as exc:
        raise ConfigError(f"{path}: {exc}") from exc
    root = _Table(data, path.name)

    run = root.table("run")
    warmup_days = run.integer("warmup_days", low=1)
    scored_days = run.integer("scored_days", low=1)
    run.close()

    traffic_t = root.table("traffic")
    traffic = Traffic(
        rate_scale=traffic_t.number("rate_scale", low=0.0),
        # Below 1 so the diurnal curve never drives a rate negative.
        diurnal_amplitude=traffic_t.number("diurnal_amplitude", low=0.0, high=0.99),
        diurnal_peak_hour=traffic_t.number("diurnal_peak_hour", low=0.0, high=24.0),
        weekend_factor=traffic_t.number("weekend_factor", low=0.0),
    )
    traffic_t.close()

    providers = tuple(_provider(t) for t in root.tables("providers"))
    if not providers:
        raise ConfigError("providers: at least one is required")
    if len({p.name for p in providers}) != len(providers):
        raise ConfigError("providers: names must be unique")

    archetypes_t = root.table("keys")
    archetypes = {name: _archetype(archetypes_t.table(name), name) for name in ARCHETYPES}
    archetypes_t.close()

    orgs = _orgs(root.table("orgs"))

    bursts_t = root.table("bursts")
    bursts = Bursts(
        rate_per_second_median=bursts_t.number("rate_per_second_median", low=0.0),
        rate_log_sigma=bursts_t.number("rate_log_sigma", low=0.0),
        duration_seconds_median=bursts_t.number("duration_seconds_median", low=1.0),
        duration_log_sigma=bursts_t.number("duration_log_sigma", low=0.0),
        duration_seconds_max=bursts_t.number("duration_seconds_max", low=1.0),
    )
    bursts_t.close()

    exports_t = root.table("exports")
    exports = Exports(
        part_bytes=exports_t.integer("part_bytes", low=1),
        part_interval_seconds=exports_t.range("part_interval_seconds", low=0.001),
        ad_hoc_per_day=exports_t.number("ad_hoc_per_day", low=0.0),
        ad_hoc_bytes_median=exports_t.number("ad_hoc_bytes_median", low=1.0),
        ad_hoc_bytes_log_sigma=exports_t.number("ad_hoc_bytes_log_sigma", low=0.0),
        scheduled_share=exports_t.number("scheduled_share", low=0.0, high=1.0),
        scheduled_bytes_median=exports_t.number("scheduled_bytes_median", low=1.0),
        scheduled_bytes_log_sigma=exports_t.number("scheduled_bytes_log_sigma", low=0.0),
        scheduled_local_hour=exports_t.range("scheduled_local_hour", low=0.0, high=24.0),
        scheduled_jitter_minutes=exports_t.number("scheduled_jitter_minutes", low=0.0),
    )
    exports_t.close()

    health_t = root.table("health")
    health = Health(
        interval_seconds=health_t.integer("interval_seconds", low=1),
        latency_ms_median=health_t.number("latency_ms_median", low=0.0),
        latency_log_sigma=health_t.number("latency_log_sigma", low=0.0),
    )
    health_t.close()

    errors_t = root.table("errors")
    background = []
    for t in errors_t.tables("background"):
        background.append(BackgroundError(_template(t), t.number("rate_per_hour", low=0.0)))
        t.close()
    novel = []
    for t in errors_t.tables("novel"):
        novel.append(_template(t))
        t.close()
    errors_t.close()
    texts = [b.template.text for b in background] + [n.text for n in novel]
    if len(set(texts)) != len(texts):
        raise ConfigError("errors: every template text must be unique, or a novel error is not novel")

    scenarios = _scenarios(root.table("scenarios"))
    if scenarios.novel_error > len(novel):
        raise ConfigError(
            f"scenarios.novel_error: {scenarios.novel_error} scenarios need as many errors.novel "
            f"templates, found {len(novel)}; each template is used once so it stays novel"
        )

    root.close()
    return Config(
        sha256=hashlib.sha256(raw).hexdigest(),
        warmup_days=warmup_days,
        scored_days=scored_days,
        traffic=traffic,
        providers=providers,
        orgs=orgs,
        archetypes=archetypes,
        bursts=bursts,
        exports=exports,
        health=health,
        background_errors=tuple(background),
        novel_errors=tuple(novel),
        scenarios=scenarios,
    )


def _timezone(name: str, path: str) -> str:
    try:
        zone(name)
    except (ValueError, OSError) as exc:
        raise ConfigError(f"{path}: unknown timezone {name!r}") from exc
    return name


def _provider(t: _Table) -> Provider:
    name = t.text("name")
    weight = t.number("weight", low=0.0)
    maintenance = None
    if t.has("maintenance"):
        m = t.table("maintenance")
        maintenance = Maintenance(
            timezone=_timezone(m.text("timezone"), f"providers.{name}.maintenance.timezone"),
            start_hour=m.range("start_hour", low=0.0, high=24.0),
            duration_minutes=m.range("duration_minutes", low=0.0),
            jitter_minutes=m.number("jitter_minutes", low=0.0),
        )
        m.close()
    t.close()
    return Provider(name, weight, maintenance)


def _archetype(t: _Table, name: str) -> KeyArchetype:
    archetype = KeyArchetype(
        weight=t.number("weight", low=0.0),
        rate_per_second=t.range("rate_per_second", low=0.0),
        bursts_per_day=t.number("bursts_per_day", low=0.0) if name == "bursty" else 0.0,
        gamma_shape=t.range("gamma_shape", low=0.01) if name == "overdispersed" else None,
    )
    t.close()
    return archetype


def _orgs(t: _Table) -> Orgs:
    keys_per_org = t.range("keys_per_org", low=1)
    if keys_per_org.low != int(keys_per_org.low) or keys_per_org.high != int(keys_per_org.high):
        raise ConfigError("orgs.keys_per_org: expected integers")
    timezones = tuple(_timezone(name, "orgs.timezones") for name in t.texts("timezones"))
    fixed = []
    for f in t.tables("fixed"):
        keys = f.texts("keys")
        unknown = sorted(set(keys) - set(ARCHETYPES))
        if unknown:
            raise ConfigError(f"orgs.fixed: unknown key archetypes {unknown}")
        fixed.append(FixedOrg(keys))
        f.close()
    orgs = Orgs(
        count=t.integer("count", low=0),
        keys_per_org=(int(keys_per_org.low), int(keys_per_org.high)),
        timezones=timezones,
        fixed=tuple(fixed),
    )
    t.close()
    return orgs


def _template(t: _Table) -> MessageTemplate:
    text = t.text("template")
    try:
        fields = {name for _, name, _, _ in string.Formatter().parse(text) if name is not None}
    except ValueError as exc:
        raise ConfigError(f"error template {text!r}: {exc}") from exc
    if not fields <= TEMPLATE_FIELDS:
        raise ConfigError(
            f"error template {text!r}: fields {sorted(fields - TEMPLATE_FIELDS)} not in {sorted(TEMPLATE_FIELDS)}"
        )
    severity = t.text("severity")
    if severity not in SEVERITIES:
        raise ConfigError(f"error template {text!r}: severity must be one of {SEVERITIES}")
    return MessageTemplate(service=t.text("service"), severity=severity, text=text)


def _scenarios(t: _Table) -> Scenarios:
    scenarios = Scenarios(
        burst_on_steady_key=t.integer("burst_on_steady_key", low=0),
        unplanned_outage=t.integer("unplanned_outage", low=0),
        maintenance_overrun=t.integer("maintenance_overrun", low=0),
        novel_error=t.integer("novel_error", low=0),
        unscheduled_bulk_export=t.integer("unscheduled_bulk_export", low=0),
        key_silence=t.integer("key_silence", low=0),
        onboarding=t.integer("onboarding", low=0),
        min_gap_hours=t.number("min_gap_hours", low=0.0),
        outage_minutes=t.range("outage_minutes", low=0.0),
        outage_local_hour=t.range("outage_local_hour", low=0.0, high=24.0),
        overrun_minutes=t.range("overrun_minutes", low=0.0),
        novel_error_count=t.integer("novel_error_count", low=1),
        novel_error_window_minutes=t.number("novel_error_window_minutes", low=0.0),
        silence_hours=t.number("silence_hours", low=0.0),
        silence_local_hour=t.range("silence_local_hour", low=0.0, high=24.0),
        silence_min_rate_per_second=t.number("silence_min_rate_per_second", low=0.0),
        onboarding_ramp_days=t.number("onboarding_ramp_days", low=0.0),
    )
    t.close()
    return scenarios
