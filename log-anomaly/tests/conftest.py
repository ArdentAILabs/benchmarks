from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from logsim.config import Config, load
from logsim.generate import Record, generate_chunk
from logsim.plan import NS, Plan, build_plan
from logsim.world import World, build_world

SMALL = Path(__file__).with_name("small.toml")
# Seed 2 puts orgs on both the Berlin-scheduled and the UTC-scheduled provider,
# which the DST tests need.
SEED = 2
START = int(datetime(2026, 10, 23, tzinfo=UTC).timestamp()) * NS


def variant(tmp_path: Path, replacements: dict[str, str]) -> Path:
    """The small config with exact text replacements; each must match once."""
    text = SMALL.read_text()
    for old, new in replacements.items():
        assert text.count(old) == 1, old
        text = text.replace(old, new)
    path = tmp_path / "variant.toml"
    path.write_text(text)
    return path


def run(cfg: Config, seed: int = SEED, start: int = START) -> tuple[World, Plan, list[Record]]:
    world = build_world(cfg, seed)
    plan = build_plan(cfg, world, seed, start)
    records = [r for i in range(plan.chunks) for r in generate_chunk(cfg, world, plan, seed, i)]
    return world, plan, records


@pytest.fixture(scope="session")
def cfg() -> Config:
    return load(SMALL)


@pytest.fixture(scope="session")
def small_run(cfg: Config) -> tuple[World, Plan, list[Record]]:
    return run(cfg)
