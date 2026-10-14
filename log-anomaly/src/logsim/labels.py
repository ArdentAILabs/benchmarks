"""Quiet cases labelled from the emitted error lines, after the plan is built.

Both are look-alikes of a novel_error, which is `novel_error_count` lines of a
never-seen message within `novel_error_window_minutes`:

- chance_error_clump: background errors arrive as independent Poisson
  processes, so that shape turns up by chance for messages the fleet logs every
  day. Each run of at least that many lines of one message within that window
  is labelled, merged where runs overlap.
- renamed_error: a deploy renames a background message without changing how
  often it is logged. Its new text has never been seen, but nothing is wrong.
  The case runs from the deploy to the new text's `novel_error_count`-th line,
  after which it is no longer new.

A detector that pages on volume or novelty alone is charged for both.
"""

from __future__ import annotations

from collections import defaultdict

from logsim.config import Config
from logsim.generate import background_errors, background_text
from logsim.plan import HOUR, MINUTE, Plan, Truth
from logsim.world import World


def chance_error_clumps(cfg: Config, world: World, plan: Plan, seed: int) -> tuple[Truth, ...]:
    need = cfg.scenarios.novel_error_count
    window = int(cfg.scenarios.novel_error_window_minutes * MINUTE)
    by_text: dict[str, list[tuple[int, int]]] = defaultdict(list)
    for index in range(plan.chunks):
        for t, template, key in background_errors(cfg, world, plan, seed, index):
            by_text[background_text(cfg, plan, template, t).text].append((t, key))

    runs: list[tuple[str, list[tuple[int, int]]]] = []  # (text, events), overlapping runs merged
    for text, events in sorted(by_text.items()):
        events.sort()
        current: list[tuple[int, int]] = []
        last = -1  # index into events of the open run's last event
        for i in range(len(events) - need + 1):
            j = i + need - 1
            if events[j][0] - events[i][0] > window:
                continue
            if current and i <= last:
                current.extend(events[last + 1 : j + 1])
            else:
                current = list(events[i : j + 1])
                runs.append((text, current))
            last = j
    clumps = []
    for text, events in sorted(runs, key=lambda r: (r[1][0][0], r[0])):
        clumps.append(
            Truth(
                id=f"c{len(clumps) + 1:05d}",
                kind="chance_error_clump",
                pair="error",
                expect="quiet",
                org=None,
                key=None,
                start=events[0][0],
                end=events[-1][0],
                scored=events[0][0] >= plan.scored_from,
                detail={"template": text, "count": len(events), **_members(world, events)},
            )
        )
    return tuple(clumps)


def renamed_errors(cfg: Config, world: World, plan: Plan, seed: int) -> tuple[Truth, ...]:
    need = cfg.scenarios.novel_error_count
    cases = []
    for number, rename in enumerate(plan.renames, start=1):
        sightings: list[tuple[int, int]] = []
        for index in range((rename.time - plan.start) // HOUR, plan.chunks):
            hour = sorted(
                (t, key)
                for t, template, key in background_errors(cfg, world, plan, seed, index)
                if template == rename.template and t >= rename.time
            )
            sightings.extend(hour[: need - len(sightings)])
            if len(sightings) == need:
                break
        # Fewer sightings than that by the end of the run: the case is still
        # open, and ending it at the run's end keeps it out of scoring.
        end = sightings[-1][0] if len(sightings) == need else plan.end
        original = cfg.background_errors[rename.template].template.text
        cases.append(
            Truth(
                id=f"r{number:05d}",
                kind="renamed_error",
                pair="error",
                expect="quiet",
                org=None,
                key=None,
                start=rename.time,
                end=end,
                scored=rename.time >= plan.scored_from,
                detail={"from": original, "to": rename.to.text, "count": len(sightings), **_members(world, sightings)},
            )
        )
    return tuple(cases)


def _members(world: World, events: list[tuple[int, int]]) -> dict[str, list[str]]:
    return {
        "keys": sorted({world.keys[k].id for _, k in events}),
        "orgs": sorted({world.orgs[world.keys[k].org].id for _, k in events}),
    }
