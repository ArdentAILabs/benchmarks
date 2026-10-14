"""Draw the simulated tenants: orgs, their API keys, and their database providers."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from logsim.config import ARCHETYPES, Config, Provider
from logsim.repro import Stream, rng


@dataclass(frozen=True)
class WeeklyExport:
    weekday: int  # org-local, 0 = Monday
    minute_of_day: int  # org-local


@dataclass(frozen=True)
class Key:
    index: int  # position in World.keys; part of every per-key random stream
    id: str
    org: int
    archetype: str
    rate: float  # requests/s on an average weekday, before rate_scale
    gamma_shape: float | None
    weekly_export: WeeklyExport | None
    batch_minute: int | None  # org-local minute of day a daily batch job starts


@dataclass(frozen=True)
class Org:
    index: int
    id: str
    timezone: str
    provider: Provider
    maintenance_minute: int | None  # provider-local minute of day the window opens
    health_phase_ms: int
    fixed: bool
    keys: tuple[int, ...]


@dataclass(frozen=True)
class World:
    orgs: tuple[Org, ...]
    keys: tuple[Key, ...]


def build_world(cfg: Config, seed: int) -> World:
    draw = rng(seed, Stream.WORLD)
    used_ids: set[str] = set()
    orgs: list[Org] = []
    keys: list[Key] = []
    archetype_p = _probabilities([cfg.archetypes[a].weight for a in ARCHETYPES], "keys.*.weight")
    provider_p = _probabilities([p.weight for p in cfg.providers], "providers.weight")

    plans = [(True, f.keys) for f in cfg.orgs.fixed] + [(False, None)] * cfg.orgs.count
    for fixed, archetypes in plans:
        org_index = len(orgs)
        timezone = cfg.orgs.timezones[int(draw.integers(len(cfg.orgs.timezones)))]
        provider = cfg.providers[int(draw.choice(len(cfg.providers), p=provider_p))]
        maintenance_minute = None
        if provider.maintenance is not None:
            hours = provider.maintenance.start_hour
            maintenance_minute = int(draw.integers(int(hours.low * 60), int(hours.high * 60)))
        health_phase_ms = int(draw.integers(cfg.health.interval_seconds * 1000))
        if archetypes is None:
            low, high = cfg.orgs.keys_per_org
            count = int(draw.integers(low, high + 1))
            archetypes = tuple(ARCHETYPES[int(draw.choice(len(ARCHETYPES), p=archetype_p))] for _ in range(count))
        org_keys = []
        for archetype in archetypes:
            spec = cfg.archetypes[archetype]
            index = len(keys)
            # Its own stream, so adding daily batches left every other draw of
            # the world where it was.
            batch = rng(seed, Stream.BATCHES, index)
            batch_minute = None
            if batch.random() < cfg.batches.share:
                low, high = int(cfg.batches.local_hour.low * 60), int(cfg.batches.local_hour.high * 60)
                batch_minute = int(batch.integers(low, max(high, low + 1)))
            weekly = None
            if draw.random() < cfg.exports.scheduled_share:
                hours = cfg.exports.scheduled_local_hour
                weekly = WeeklyExport(
                    weekday=int(draw.integers(7)),
                    minute_of_day=int(draw.integers(int(hours.low * 60), int(hours.high * 60))),
                )
            key = Key(
                index=index,
                id=_unique_id(draw, "key", used_ids),
                org=org_index,
                archetype=archetype,
                rate=float(draw.uniform(spec.rate_per_second.low, spec.rate_per_second.high)),
                gamma_shape=(
                    float(draw.uniform(spec.gamma_shape.low, spec.gamma_shape.high))
                    if spec.gamma_shape is not None
                    else None
                ),
                weekly_export=weekly,
                batch_minute=batch_minute,
            )
            keys.append(key)
            org_keys.append(key.index)
        orgs.append(
            Org(
                index=org_index,
                id=_unique_id(draw, "org", used_ids),
                timezone=timezone,
                provider=provider,
                maintenance_minute=maintenance_minute,
                health_phase_ms=health_phase_ms,
                fixed=fixed,
                keys=tuple(org_keys),
            )
        )
    return World(orgs=tuple(orgs), keys=tuple(keys))


def _probabilities(weights: list[float], path: str) -> np.ndarray:
    total = sum(weights)
    if total <= 0:
        raise ValueError(f"{path}: weights must not all be zero")
    return np.asarray(weights) / total


def _unique_id(draw: np.random.Generator, prefix: str, used: set[str]) -> str:
    value = f"{prefix}_{int(draw.integers(1 << 40)):010x}"
    if value in used:
        # About 1e-9 for a hundred ids at 40 bits; refuse rather than carry a
        # retry path that no test or run will ever exercise.
        raise ValueError(f"id collision on {value}; choose another seed")
    used.add(value)
    return value
