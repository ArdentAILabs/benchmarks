"""The two inputs that make a run reproducible: random streams and timezone rules.

Every random draw comes from a stream keyed by (seed, purpose, entity, chunk),
so any hour of any entity can be regenerated on its own. That is what lets a
live run restart mid-way and emit exactly what it would have emitted.
"""

from __future__ import annotations

import functools
import importlib.metadata
import importlib.resources
import platform
from enum import IntEnum
from zoneinfo import ZoneInfo

import numpy as np


class Stream(IntEnum):
    # Values are part of the output's identity: renumbering one changes every
    # run that uses it. Append; never reorder.
    WORLD = 1
    BURSTS = 2
    MAINTENANCE = 3
    EXPORTS = 4
    SCENARIOS = 5
    REQUESTS = 6
    HEALTH = 7
    ERRORS = 8
    NOVEL_ERRORS = 9


def rng(seed: int, stream: Stream, *parts: int) -> np.random.Generator:
    if seed < 0:
        raise ValueError("seed must be non-negative")
    sequence = np.random.SeedSequence(seed, spawn_key=(int(stream), *parts))
    return np.random.Generator(np.random.PCG64(sequence))


@functools.cache
def zone(name: str) -> ZoneInfo:
    """Load a zone from the pinned tzdata package, never the host's database, so
    DST rules are identical on every machine that installs the lockfile."""
    resource = importlib.resources.files("tzdata.zoneinfo").joinpath(*name.split("/"))
    if not resource.is_file():
        raise ValueError(f"unknown timezone {name!r}")
    with resource.open("rb") as handle:
        return ZoneInfo.from_file(handle, key=name)


def versions() -> dict[str, str]:
    return {
        "logsim": importlib.metadata.version("logsim"),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "tzdata": importlib.metadata.version("tzdata"),
    }
