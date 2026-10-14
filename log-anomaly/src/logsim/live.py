"""Emit a plan in real time, for systems that only analyse data as it arrives.

A restart regenerates the same plan and resumes at the current wall-clock
second; whatever fell between the crash and the restart is never sent. The
journal records each start, so a scorer can exclude those gaps rather than
count them against a detector.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from pathlib import Path
from typing import Protocol, TextIO

from logsim.config import Config
from logsim.generate import Record, generate_chunk
from logsim.output import iso
from logsim.plan import HOUR, MINUTE, NS, Plan
from logsim.world import World

# Records reaching a detector this late have stopped being live; stop loudly
# instead of feeding both systems a backlog they will treat differently.
MAX_LAG = MINUTE


class Sink(Protocol):
    def send(self, records: list[Record], observed: int) -> None: ...


class FellBehind(RuntimeError):
    pass


def stream(
    cfg: Config,
    world: World,
    plan: Plan,
    seed: int,
    sink: Sink,
    journal_path: Path,
    clock: Callable[[], int] = time.time_ns,
    sleep: Callable[[float], None] = time.sleep,
) -> None:
    now = clock()
    if now >= plan.end:
        raise RuntimeError(f"the run ended at {iso(plan.end)}")
    resume = max(now, plan.start)
    first = (resume - plan.start) // HOUR
    with journal_path.open("a", encoding="utf-8") as journal:
        _note(journal, clock, "emitting", emitting_from=iso(resume))
        sent = 0
        through = resume  # every record before this instant has been sent
        noted = clock()
        try:
            for index in range(first, plan.chunks):
                records = generate_chunk(cfg, world, plan, seed, index)
                if index == first:
                    records = [r for r in records if r.time >= resume]
                i = 0
                while i < len(records):
                    second_end = (records[i].time // NS + 1) * NS
                    j = i
                    while j < len(records) and records[j].time < second_end:
                        j += 1
                    wait = second_end - clock()
                    if wait > 0:
                        sleep(wait / NS)
                    lag = clock() - second_end
                    if lag > MAX_LAG:
                        raise FellBehind(f"{lag / NS:.1f}s behind real time at {iso(second_end)}")
                    sink.send(records[i:j], clock())
                    sent += j - i
                    through = second_end
                    i = j
                    if clock() - noted >= MINUTE:
                        _note(journal, clock, "progress", sent_through=iso(through), records=sent)
                        noted = clock()
        except BaseException as exc:
            _note(journal, clock, "stopped", error=repr(exc), sent_through=iso(through), records=sent)
            raise
        _note(journal, clock, "finished", records=sent)


def _note(journal: TextIO, clock: Callable[[], int], event: str, **fields: object) -> None:
    journal.write(json.dumps({"at": iso(clock()), "event": event, **fields}) + "\n")
    journal.flush()
