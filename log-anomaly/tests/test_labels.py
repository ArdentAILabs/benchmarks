from __future__ import annotations

import re
from itertools import pairwise
from pathlib import Path

from conftest import SEED, START, variant

from logsim.config import Config, load
from logsim.generate import Record, generate_chunk
from logsim.labels import chance_error_clumps, renamed_errors
from logsim.plan import HOUR, MINUTE, Plan, Truth, build_plan
from logsim.world import World, build_world

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
    texts = [b.template.text for b in cfg.background_errors] + [r.to.text for r in plan.renames]
    assert len(texts) > len(cfg.background_errors), "the small run should rename a message"
    for text in texts:
        pattern = _pattern(text)
        times = sorted(r.time for r in records if pattern.match(r.body))
        mine = [c for c in clumps if c.detail["template"] == text]
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


def _renamed(cfg: Config, world: World, plan: Plan, records: list[Record]) -> tuple[Truth, list[tuple[int, str]]]:
    (case,) = renamed_errors(cfg, world, plan, SEED)
    (rename,) = plan.renames
    assert (case.kind, case.pair, case.expect, case.start) == ("renamed_error", "error", "quiet", rename.time)
    assert case.detail["from"] == cfg.background_errors[rename.template].template.text
    pattern = _pattern(rename.to.text)
    return case, sorted((r.time, r.attributes["api_key.id"]) for r in records if pattern.match(r.body))


def test_a_renamed_error_ends_when_its_new_text_is_no_longer_new(tmp_path: Path) -> None:
    """With the renamed message logged often, the case closes on its
    novel_error_count-th line and names the keys that logged them."""
    busy = load(
        variant(
            tmp_path,
            {
                'template = "connection pool exhausted after {ms}ms wait"\nrate_per_hour = 1.0': (
                    'template = "connection pool exhausted after {ms}ms wait"\nrate_per_hour = 100.0'
                )
            },
        )
    )
    world = build_world(busy, SEED)
    plan = build_plan(busy, world, SEED, START)
    (rename,) = plan.renames
    assert busy.background_errors[rename.template].rate_per_hour == 100.0
    first = (rename.time - plan.start) // HOUR
    records = [r for i in range(first, first + 2) for r in generate_chunk(busy, world, plan, SEED, i)]
    case, sightings = _renamed(busy, world, plan, records)
    need = busy.scenarios.novel_error_count
    assert case.end == sightings[need - 1][0]
    assert case.detail["keys"] == sorted({k for _, k in sightings[:need]})
    # Chance clumps follow the text as logged: after the rename they are the
    # new text's, and none of the old text's straddles the deploy.
    old = busy.background_errors[rename.template].template.text
    clumps = chance_error_clumps(busy, world, plan, SEED)
    assert any(c.detail["template"] == rename.to.text for c in clumps)
    assert all(c.end < rename.time for c in clumps if c.detail["template"] == old)


def test_a_renamed_error_still_new_at_the_end_stays_open(cfg: Config, small_run: Run) -> None:
    world, plan, records = small_run
    case, sightings = _renamed(cfg, world, plan, records)
    assert 0 < len(sightings) < cfg.scenarios.novel_error_count, "the small run's rename should still be new"
    # Ending at the run's end keeps it out of scoring: it never finished.
    assert case.end == plan.end and case.detail["count"] == len(sightings)
