"""Label chance clumps of known errors: the quiet half of the novel_error pair.

A novel_error case is `novel_error_count` lines of a never-seen message within
`novel_error_window_minutes`. Background errors arrive as independent Poisson
processes, so the same shape turns up by chance for messages the fleet logs
every day. Each run of at least that many lines of one background template
within that window is labelled here, merged where runs overlap, so a detector
that pages on error volume alone is charged for it.
"""

from __future__ import annotations

from collections import defaultdict

from logsim.config import Config
from logsim.generate import background_errors
from logsim.plan import MINUTE, Plan, Truth
from logsim.world import World


def chance_error_clumps(cfg: Config, world: World, plan: Plan, seed: int) -> tuple[Truth, ...]:
    need = cfg.scenarios.novel_error_count
    window = int(cfg.scenarios.novel_error_window_minutes * MINUTE)
    by_template: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for index in range(plan.chunks):
        for t, template, key in background_errors(cfg, world, plan, seed, index):
            by_template[template].append((t, key))

    runs: list[tuple[int, list[tuple[int, int]]]] = []  # (template, events), overlapping runs merged
    for template, events in sorted(by_template.items()):
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
                runs.append((template, current))
            last = j
    clumps = []
    for template, events in sorted(runs, key=lambda r: (r[1][0][0], r[0])):
        keys = sorted({world.keys[k].id for _, k in events})
        orgs = sorted({world.orgs[world.keys[k].org].id for _, k in events})
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
                detail={
                    "template": cfg.background_errors[template].template.text,
                    "count": len(events),
                    "keys": keys,
                    "orgs": orgs,
                },
            )
        )
    return tuple(clumps)
