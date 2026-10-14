from __future__ import annotations

import re
from itertools import pairwise

from conftest import SEED

from logsim.clumps import chance_error_clumps
from logsim.config import Config
from logsim.generate import Record
from logsim.plan import MINUTE, Plan
from logsim.world import World

Run = tuple[World, Plan, list[Record]]


def _pattern(template: str) -> re.Pattern[str]:
    return re.compile(re.sub(r"\\\{(ms|id|n)\\\}", ".+", re.escape(template)) + "$")


def test_every_chance_clump_is_found_and_nothing_else(cfg: Config, small_run: Run) -> None:
    """Brute force over the emitted records: every window holding enough lines
    of one background template lies inside exactly one labelled clump, and each
    clump is made of such windows."""
    world, plan, records = small_run
    clumps = chance_error_clumps(cfg, world, plan, SEED)
    assert clumps, "the small run should clump by chance"
    need = cfg.scenarios.novel_error_count
    window = int(cfg.scenarios.novel_error_window_minutes * MINUTE)
    for background in cfg.background_errors:
        pattern = _pattern(background.template.text)
        times = sorted(r.time for r in records if pattern.match(r.body))
        mine = [c for c in clumps if c.detail["template"] == background.template.text]
        covered = set()
        for i in range(len(times) - need + 1):
            if times[i + need - 1] - times[i] <= window:
                (holder,) = [c for c in mine if c.start <= times[i] and times[i + need - 1] <= c.end]
                covered.add(holder.id)
        assert covered == {c.id for c in mine}
        # Overlapping windows merge into one case, so one burst of errors is
        # one false page at most, not one per sliding window.
        spans = sorted((c.start, c.end) for c in mine)
        assert all(a_end < b_start for (_, a_end), (b_start, _) in pairwise(spans))
        for clump in mine:
            inside = [t for t in times if clump.start <= t <= clump.end]
            assert len(inside) == clump.detail["count"]


def test_clumps_are_quiet_fleet_wide_cases_naming_their_members(cfg: Config, small_run: Run) -> None:
    world, plan, _ = small_run
    key_ids = {k.id for k in world.keys}
    for clump in chance_error_clumps(cfg, world, plan, SEED):
        assert (clump.kind, clump.pair, clump.expect, clump.org, clump.key) == (
            "chance_error_clump",
            "error",
            "quiet",
            None,
            None,
        )
        assert clump.detail["count"] >= cfg.scenarios.novel_error_count
        assert set(clump.detail["keys"]) <= key_ids
        assert clump.detail["template"] not in {n.text for n in cfg.novel_errors}
