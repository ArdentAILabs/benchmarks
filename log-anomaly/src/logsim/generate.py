"""Turn one hour of a plan into log records.

Each hour draws from its own random streams, so hour N is identical whether it
is generated alone or after hours 0..N-1. Natural and injected events go
through the same record builders: nothing in a record's shape says which kind
produced it.
"""

from __future__ import annotations

import bisect
import math
from collections.abc import Iterable
from typing import NamedTuple

import numpy as np

from logsim.config import Config, MessageTemplate
from logsim.plan import HOUR, NS, OVERLOADED, SERVER_ERROR, Plan, Ramp
from logsim.repro import Stream, rng, zone
from logsim.traffic import profile
from logsim.world import Key, Org, World

Attribute = str | int | float

# (method, route, success status, client-error status, share of requests)
_ROUTES = (
    ("POST", "/v1/branches", 201, 409, 0.70),
    ("GET", "/v1/branches/{branch_id}", 200, 404, 0.25),
    ("DELETE", "/v1/branches/{branch_id}", 204, 404, 0.05),
)
_ROUTE_P = np.array([r[4] for r in _ROUTES])
_OK_STATUS = np.array([r[2] for r in _ROUTES])
_CLIENT_STATUS = np.array([r[3] for r in _ROUTES])
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
    records.extend(_retries(cfg, world, plan, seed, index, start, end))
    records.extend(_deploys(plan, start, end))
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

    rate = key.rate * scale * profile(seconds, zone(org.timezone), cfg.traffic)
    if key.gamma_shape is not None:
        # Gamma-mixed Poisson per minute: a negative binomial count, overdispersed
        # with variance mean + mean^2 / shape.
        rate *= np.repeat(draw.gamma(key.gamma_shape, 1.0 / key.gamma_shape, 60), 60)
    rate *= _ramp(plan.rate_ramps.get(key.org, ()), instants)
    for surge in plan.surges.get(key.index, ()):
        rate[(instants >= surge.start) & (instants < surge.end)] *= surge.multiplier
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
    second = np.repeat(np.arange(3600), counts)
    times = instants[second] + draw.integers(0, NS, total)
    routes = draw.choice(len(_ROUTES), size=total, p=_ROUTE_P)
    outcome = draw.random(total)
    latency = draw.lognormal(math.log(_REQUEST_MS_MEDIAN), _REQUEST_MS_SIGMA, total)
    request_ids = draw.integers(0, 1 << 63, total)

    share = np.full(3600, cfg.traffic.server_error_share)
    status = np.full(3600, SERVER_ERROR)
    for span in plan.error_spans.get(key.index, ()):
        inside = (instants >= span.start) & (instants < span.end) & (span.share >= share)
        share[inside] = span.share
        status[inside] = span.status
    server = outcome < share[second]
    rejected = ~server & (outcome < share[second] + _CLIENT_ERROR_SHARE)
    codes = np.where(server, status[second], np.where(rejected, _CLIENT_STATUS[routes], _OK_STATUS[routes]))
    latency *= _ramp(plan.latency_ramps.get(key.org, ()), times)
    return _render(org, key, times, routes, codes, latency, request_ids)


def _retries(cfg: Config, world: World, plan: Plan, seed: int, index: int, start: int, end: int) -> list[Record]:
    """Retries are requests like any other: same routes, attributes and bodies.
    They fail with 503 while their dependency is overloaded and succeed after."""
    records: list[Record] = []
    for number, cascade in enumerate(plan.cascades):
        last = cascade.start + len(cascade.attempts) * NS
        if cascade.start >= end or last <= start:
            continue
        key = world.keys[cascade.key]
        first = cascade.start // NS
        offsets = np.arange(len(cascade.attempts))
        instants = (first + offsets) * NS
        inside = (instants >= start) & (instants < end)
        counts = np.asarray(cascade.attempts)[inside]
        total = int(counts.sum())
        if total == 0:
            continue
        draw = rng(seed, Stream.RETRY_RECORDS, number, index)
        times = np.repeat(instants[inside], counts) + draw.integers(0, NS, total)
        routes = draw.choice(len(_ROUTES), size=total, p=_ROUTE_P)
        latency = draw.lognormal(math.log(_REQUEST_MS_MEDIAN), _REQUEST_MS_SIGMA, total)
        request_ids = draw.integers(0, 1 << 63, total)
        codes = np.where(times < cascade.overload_end, OVERLOADED, _OK_STATUS[routes])
        latency *= _ramp(plan.latency_ramps.get(key.org, ()), times)
        records.extend(_render(world.orgs[key.org], key, times, routes, codes, latency, request_ids))
    return records


def _render(
    org: Org,
    key: Key,
    times: np.ndarray,
    routes: np.ndarray,
    codes: np.ndarray,
    latency: np.ndarray,
    request_ids: np.ndarray,
) -> list[Record]:
    records = []
    milliseconds = np.rint(latency).astype(np.int64)
    for t, r, status, ms, rid in zip(
        times.tolist(), routes.tolist(), codes.tolist(), milliseconds.tolist(), request_ids.tolist(), strict=True
    ):
        method, route = _ROUTES[r][:2]
        records.append(
            Record(
                t,
                "ERROR" if status >= 500 else "WARN" if status >= 400 else "INFO",
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


def _ramp(ramps: Iterable[Ramp], times: np.ndarray) -> np.ndarray:
    """The product of every ramp's multiplier at each time."""
    factor = np.ones(len(times))
    for ramp in ramps:
        progress = np.clip((times - ramp.start) / (ramp.end - ramp.start), 0.0, 1.0)
        rising = 1.0 + (ramp.factor - 1.0) * progress
        live = times >= ramp.start if ramp.holds else (times >= ramp.start) & (times < ramp.end)
        factor *= np.where(live, rising, 1.0)
    return factor


def _deploys(plan: Plan, start: int, end: int) -> list[Record]:
    return [
        Record(d.time, "INFO", "deployer", f"deployed release {d.release}", {"deploy.release": d.release})
        for d in plan.deploys
        if start <= d.time < end
    ]


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


def background_text(cfg: Config, plan: Plan, template: int, t: int) -> MessageTemplate:
    """The text a background error is logged with at `t`: its configured
    template, or what a deploy before `t` renamed it to."""
    renamed = [r for r in plan.renames if r.template == template]
    position = bisect.bisect_right([r.time for r in renamed], t)
    return renamed[position - 1].to if position else cfg.background_errors[template].template


def _errors(cfg: Config, world: World, plan: Plan, seed: int, index: int, start: int, end: int) -> list[Record]:
    records = []
    events = background_errors(cfg, world, plan, seed, index)
    if events:
        text = rng(seed, Stream.ERROR_TEXT, index)
        for t, template, key in events:
            records.append(_error(world, world.keys[key], background_text(cfg, plan, template, t), t, text))
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
