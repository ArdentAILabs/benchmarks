"""Turn one hour of a plan into log records.

Each hour draws from its own random streams, so hour N is identical whether it
is generated alone or after hours 0..N-1. Natural and injected events go
through the same record builders: nothing in a record's shape says which kind
produced it.
"""

from __future__ import annotations

import math
from datetime import UTC, datetime
from typing import NamedTuple
from zoneinfo import ZoneInfo

import numpy as np

from logsim.config import Config, MessageTemplate, Traffic
from logsim.plan import HOUR, NS, Plan
from logsim.repro import Stream, rng, zone
from logsim.world import Key, Org, World

Attribute = str | int | float

# (method, route, success status, client-error status, share of requests)
_ROUTES = (
    ("POST", "/v1/branches", 201, 409, 0.70),
    ("GET", "/v1/branches/{branch_id}", 200, 404, 0.25),
    ("DELETE", "/v1/branches/{branch_id}", 204, 404, 0.05),
)
_ROUTE_P = np.array([r[4] for r in _ROUTES])
_CLIENT_ERROR_SHARE = 0.01
_REQUEST_MS_MEDIAN = 80.0
_REQUEST_MS_SIGMA = 0.5
_HEALTH_OK = "source health check ok"
# Planned maintenance and unplanned outages fail identically on purpose: the
# raw signal must not reveal which one it is.
_HEALTH_FAILURES = ("source unreachable: connection refused", "source unreachable: timeout after 5000ms")


class Record(NamedTuple):
    time: int  # UTC nanoseconds
    severity: str
    service: str
    body: str
    attributes: dict[str, Attribute]


def generate_chunk(cfg: Config, world: World, plan: Plan, seed: int, index: int) -> list[Record]:
    if not 0 <= index < plan.chunks:
        raise IndexError(f"chunk {index} outside run of {plan.chunks}")
    start = plan.start + index * HOUR
    end = start + HOUR
    records: list[Record] = []
    for key in world.keys:
        records.extend(_requests(cfg, world.orgs[key.org], key, plan, seed, index, start))
    records.extend(_exports(world, plan, start, end))
    for org in world.orgs:
        records.extend(_health(cfg, org, plan, seed, index, start, end))
    records.extend(_errors(cfg, world, plan, seed, index, start, end))
    # Stable sort: ties keep generation order, which is itself deterministic.
    records.sort(key=lambda r: r.time)
    return records


def _requests(cfg: Config, org: Org, key: Key, plan: Plan, seed: int, index: int, start: int) -> list[Record]:
    draw = rng(seed, Stream.REQUESTS, key.index, index)
    seconds = np.arange(start // NS, start // NS + 3600, dtype=np.int64)
    instants = seconds * NS
    scale = cfg.traffic.rate_scale

    rate = key.rate * scale * _profile(seconds, zone(org.timezone), cfg.traffic)
    if key.gamma_shape is not None:
        # Gamma-mixed Poisson per minute: a negative binomial count, overdispersed
        # with variance mean + mean^2 / shape.
        rate *= np.repeat(draw.gamma(key.gamma_shape, 1.0 / key.gamma_shape, 60), 60)
    activation = plan.activations.get(key.org)
    arrived = instants >= (activation.start if activation is not None else plan.start)
    if activation is not None:
        ramp = max(activation.ramp_end - activation.start, 1)
        # Zero before the org arrives, rising linearly to full by ramp_end.
        rate *= np.clip((instants - activation.start) / ramp, 0.0, 1.0)
    for burst in plan.bursts.get(key.index, ()):
        # Full rate even mid-ramp, so the truth file's burst rate is the one emitted.
        inside = (instants >= burst.start) & (instants < burst.end) & arrived
        rate[inside] = np.maximum(rate[inside], burst.rate * scale)
    for silence in plan.silences.get(key.index, ()):
        rate[(instants >= silence.start) & (instants < silence.end)] = 0.0

    counts = draw.poisson(rate)
    total = int(counts.sum())
    if total == 0:
        return []
    times = np.repeat(instants, counts) + draw.integers(0, NS, total)
    routes = draw.choice(len(_ROUTES), size=total, p=_ROUTE_P)
    rejected = draw.random(total) < _CLIENT_ERROR_SHARE
    latency = np.rint(draw.lognormal(math.log(_REQUEST_MS_MEDIAN), _REQUEST_MS_SIGMA, total)).astype(np.int64)
    request_ids = draw.integers(0, 1 << 63, total)

    records = []
    for t, r, bad, ms, rid in zip(
        times.tolist(), routes.tolist(), rejected.tolist(), latency.tolist(), request_ids.tolist(), strict=True
    ):
        method, route, ok, client_error, _ = _ROUTES[r]
        status = client_error if bad else ok
        records.append(
            Record(
                t,
                "WARN" if bad else "INFO",
                "api",
                f"{method} {route} {status}",
                {
                    "org.id": org.id,
                    "api_key.id": key.id,
                    "source.provider": org.provider.name,
                    "http.request.method": method,
                    "http.route": route,
                    "http.response.status_code": status,
                    "duration_ms": ms,
                    "request.id": f"{rid:016x}",
                },
            )
        )
    return records


def _profile(seconds: np.ndarray, tz: ZoneInfo, traffic: Traffic) -> np.ndarray:
    """Relative request rate by org-local time: a daily cosine and a weekend dip."""
    offset = _utc_offset(tz, int(seconds[0]))
    if _utc_offset(tz, int(seconds[-1])) != offset:
        raise ValueError(
            f"{tz.key} changes UTC offset inside the hour at {int(seconds[0])}; chunks must not straddle one"
        )
    local = seconds + offset
    hour = (local % 86400) / 3600.0
    daily = 1.0 + traffic.diurnal_amplitude * np.cos(2.0 * np.pi * (hour - traffic.diurnal_peak_hour) / 24.0)
    weekday = (local // 86400 + 3) % 7  # 1970-01-01 was a Thursday; Monday = 0
    return daily * np.where(weekday >= 5, traffic.weekend_factor, 1.0)


def _utc_offset(tz: ZoneInfo, second: int) -> int:
    delta = datetime.fromtimestamp(second, UTC).astimezone(tz).utcoffset()
    if delta is None:
        raise ValueError(f"{tz.key} has no UTC offset")
    return int(delta.total_seconds())


def _exports(world: World, plan: Plan, start: int, end: int) -> list[Record]:
    records = []
    for export in plan.exports:
        if export.parts[0][0] >= end or export.parts[-1][0] < start:
            continue
        key = world.keys[export.key]
        org = world.orgs[key.org]
        for i, (t, size) in enumerate(export.parts, start=1):
            if start <= t < end:
                records.append(
                    Record(
                        t,
                        "INFO",
                        "api",
                        f"export part {i}/{len(export.parts)} completed",
                        {
                            "org.id": org.id,
                            "api_key.id": key.id,
                            "source.provider": org.provider.name,
                            "export.id": export.id,
                            "export.part": i,
                            "export.parts": len(export.parts),
                            "export.bytes": size,
                        },
                    )
                )
    return records


def _health(cfg: Config, org: Org, plan: Plan, seed: int, index: int, start: int, end: int) -> list[Record]:
    draw = rng(seed, Stream.HEALTH, org.index, index)
    interval = cfg.health.interval_seconds * NS
    phase = org.health_phase_ms * 10**6
    checks = np.arange(start + (phase - start) % interval, end, interval, dtype=np.int64)
    jitter = draw.integers(0, NS, len(checks))
    latency = np.rint(
        draw.lognormal(math.log(cfg.health.latency_ms_median), cfg.health.latency_log_sigma, len(checks))
    ).astype(np.int64)
    reasons = draw.integers(0, len(_HEALTH_FAILURES), len(checks))
    outages = plan.outages.get(org.index, ())

    records = []
    for t, ms, reason in zip((checks + jitter).tolist(), latency.tolist(), reasons.tolist(), strict=True):
        if not plan.active(org.index, t):
            continue
        attributes: dict[str, Attribute] = {"org.id": org.id, "source.provider": org.provider.name}
        if any(o.start <= t < o.end for o in outages):
            records.append(Record(t, "ERROR", "source-monitor", _HEALTH_FAILURES[reason], attributes))
        else:
            attributes["latency_ms"] = ms
            records.append(Record(t, "INFO", "source-monitor", _HEALTH_OK, attributes))
    return records


def background_errors(cfg: Config, world: World, plan: Plan, seed: int, index: int) -> list[tuple[int, int, int]]:
    """One hour's background errors as (time, template index, key index).

    Kept apart from rendering so the chance-clump labels and the records come
    from the same draws."""
    start = plan.start + index * HOUR
    end = start + HOUR
    # A key that sends no requests this hour logs no errors either; an error
    # from a silenced key would give the silence away.
    silenced = {s.key for spans in plan.silences.values() for s in spans if s.start < end and s.end > start}
    active = [k for k in world.keys if plan.active(k.org, start) and k.index not in silenced]
    if not cfg.background_errors or not active:
        return []
    weights = np.array([k.rate for k in active])
    if weights.sum() <= 0:
        raise ValueError("background errors need at least one active key with a positive rate")
    draw = rng(seed, Stream.ERRORS, index)
    p = weights / weights.sum()
    events = []
    for template, background in enumerate(cfg.background_errors):
        count = int(draw.poisson(background.rate_per_hour))
        times = (start + draw.integers(0, HOUR, count)).tolist()
        picks = draw.choice(len(active), size=count, p=p).tolist()
        events.extend((t, template, active[pick].index) for t, pick in zip(times, picks, strict=True))
    return events


def _errors(cfg: Config, world: World, plan: Plan, seed: int, index: int, start: int, end: int) -> list[Record]:
    records = []
    events = background_errors(cfg, world, plan, seed, index)
    if events:
        text = rng(seed, Stream.ERROR_TEXT, index)
        for t, template, key in events:
            records.append(_error(world, world.keys[key], cfg.background_errors[template].template, t, text))
    novel = [n for n in plan.novel_errors if start <= n.time < end]
    if novel:
        draw = rng(seed, Stream.NOVEL_ERRORS, index)
        for n in novel:
            records.append(_error(world, world.keys[n.key], n.template, n.time, draw))
    return records


def _error(world: World, key: Key, template: MessageTemplate, t: int, draw: np.random.Generator) -> Record:
    org = world.orgs[key.org]
    body = template.text.format(
        ms=int(draw.lognormal(math.log(2000.0), 1.0)),
        id=f"{int(draw.integers(1 << 32)):08x}",
        n=int(draw.integers(1, 65)),
    )
    attributes: dict[str, Attribute] = {
        "org.id": org.id,
        "api_key.id": key.id,
        "source.provider": org.provider.name,
    }
    return Record(t, template.severity, template.service, body, attributes)
