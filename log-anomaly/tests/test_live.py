from __future__ import annotations

import json
from pathlib import Path

import pytest
from conftest import SEED

from logsim.config import Config
from logsim.generate import Record, generate_chunk
from logsim.live import FellBehind, stream
from logsim.plan import HOUR, NS, Plan
from logsim.world import World

Run = tuple[World, Plan, list[Record]]


class _Clock:
    def __init__(self, now: int):
        self.now = now

    def __call__(self) -> int:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.now += round(seconds * NS)


class _Enough(Exception):
    pass


class _Sink:
    def __init__(self, clock: _Clock, limit: int, lag_after: int | None = None):
        self.clock = clock
        self.limit = limit
        self.lag_after = lag_after
        self.batches: list[tuple[int, list[Record]]] = []

    def send(self, records: list[Record], observed: int) -> None:
        self.batches.append((observed, records))
        if self.lag_after is not None and len(self.batches) >= self.lag_after:
            self.clock.now += 120 * NS  # a send that hangs for two minutes
        if len(self.batches) >= self.limit:
            raise _Enough


def test_resume_emits_each_second_after_it_closes(cfg: Config, small_run: Run, tmp_path: Path) -> None:
    world, plan, _ = small_run
    resume = plan.start + 30 * HOUR + 1_234_567_890
    clock = _Clock(resume)
    sink = _Sink(clock, limit=500)
    journal = tmp_path / "journal.jsonl"
    with pytest.raises(_Enough):
        stream(cfg, world, plan, SEED, sink, journal, clock=clock, sleep=clock.sleep)

    expected = [r for r in generate_chunk(cfg, world, plan, SEED, 30) if r.time >= resume]
    sent = [r for _, batch in sink.batches for r in batch]
    assert sent == expected[: len(sent)]
    for observed, batch in sink.batches:
        second = batch[0].time // NS
        assert all(r.time // NS == second for r in batch)
        assert observed >= (second + 1) * NS

    events = [json.loads(line) for line in journal.read_text().splitlines()]
    assert events[0]["event"] == "emitting" and events[0]["emitting_from"].startswith("2026-10-24T06:00:01.23")
    assert events[-1]["event"] == "stopped" and events[-1]["records"] == len(sent) - len(sink.batches[-1][1])


def test_falling_behind_stops_the_run(cfg: Config, small_run: Run, tmp_path: Path) -> None:
    world, plan, _ = small_run
    clock = _Clock(plan.start + 30 * HOUR)
    sink = _Sink(clock, limit=10_000, lag_after=3)
    with pytest.raises(FellBehind, match="behind real time"):
        stream(cfg, world, plan, SEED, sink, tmp_path / "journal.jsonl", clock=clock, sleep=clock.sleep)


def test_a_finished_run_refuses_to_start(cfg: Config, small_run: Run, tmp_path: Path) -> None:
    world, plan, _ = small_run
    clock = _Clock(plan.end)
    with pytest.raises(RuntimeError, match="the run ended"):
        stream(cfg, world, plan, SEED, _Sink(clock, 1), tmp_path / "journal.jsonl", clock=clock, sleep=clock.sleep)
