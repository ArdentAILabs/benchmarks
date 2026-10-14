from __future__ import annotations

from collections import Counter
from datetime import UTC, datetime
from itertools import pairwise
from pathlib import Path

import pytest
from conftest import SEED, START, variant

from logsim.config import Config, load
from logsim.plan import DAY, HOUR, MINUTE, NS, Plan, PlanError, build_plan
from logsim.world import World, build_world


def _plan(cfg: Config, seed: int = SEED) -> tuple[World, Plan]:
    world = build_world(cfg, seed)
    return world, build_plan(cfg, world, seed, START)


def test_plan_is_a_pure_function_of_config_seed_and_start(cfg: Config) -> None:
    assert _plan(cfg)[1] == _plan(cfg)[1]
    assert _plan(cfg)[1].truth != _plan(cfg, seed=SEED + 1)[1].truth


def test_start_must_fall_on_a_utc_hour(cfg: Config) -> None:
    world = build_world(cfg, SEED)
    with pytest.raises(PlanError, match="UTC hour"):
        build_plan(cfg, world, SEED, START + 30 * MINUTE)


def test_every_injected_case_sits_inside_the_scored_phase(cfg: Config) -> None:
    _, plan = _plan(cfg)
    injected = [t for t in plan.truth if t.kind not in ("natural_burst", "maintenance_window", "scheduled_export")]
    assert len(injected) == 13  # the small config's scenario counts
    for truth in injected:
        assert plan.scored_from <= truth.start < truth.end <= plan.end, truth


def test_alerts_have_matched_quiet_cases(cfg: Config) -> None:
    _, plan = _plan(cfg)
    by_pair = Counter((t.pair, t.expect) for t in plan.truth if t.scored)
    for pair in ("burst", "outage", "egress"):
        assert by_pair[(pair, "alert")] > 0 and by_pair[(pair, "quiet")] > 0, pair


def test_no_target_repeats_within_a_kind_while_others_remain(cfg: Config) -> None:
    _, plan = _plan(cfg)
    targets = Counter((t.kind, t.key if t.key is not None else t.org) for t in plan.truth if t.expect == "alert")
    assert max(targets.values()) == 1


def test_scenarios_in_one_org_keep_their_distance(cfg: Config) -> None:
    _, plan = _plan(cfg)
    gap = int(cfg.scenarios.min_gap_hours * HOUR)
    injected = sorted(
        (t for t in plan.truth if t.kind not in ("natural_burst", "maintenance_window", "scheduled_export")),
        key=lambda t: (t.org, t.start),
    )
    for a, b in pairwise(injected):
        if a.org == b.org:
            assert b.start >= a.end + gap, (a, b)


def test_onboarding_org_is_excluded_from_other_scenarios(cfg: Config) -> None:
    _, plan = _plan(cfg)
    (newcomer,) = plan.activations
    kinds = {t.kind for t in plan.truth if t.org == newcomer and t.expect == "alert"}
    assert kinds == set()
    assert all(t.start >= plan.activations[newcomer].start for t in plan.truth if t.org == newcomer)


def test_maintenance_moves_in_utc_only_where_scheduled_in_local_time(cfg: Config) -> None:
    _, plan = _plan(cfg)
    windows = [t for t in plan.truth if t.kind == "maintenance_window"]
    berlin = [t for t in windows if t.detail["provider"] == "provider-a"]
    utc = [t for t in windows if t.detail["provider"] == "provider-c"]
    assert berlin and utc
    # Europe leaves summer time at 01:00 UTC on 2026-10-25; the next Berlin
    # window opens an hour later in UTC than the day before.
    dst_day = int(datetime(2026, 10, 25, tzinfo=UTC).timestamp()) * NS
    shifted = {t.org for t in berlin if t.detail["utc_shift_minutes"] == 60}
    assert shifted == {t.org for t in berlin}
    assert all(dst_day <= t.start < dst_day + DAY for t in berlin if t.detail["utc_shift_minutes"] == 60)
    assert all(t.detail["utc_shift_minutes"] == 0 for t in utc)


def test_overrun_replaces_the_window_it_extends(cfg: Config) -> None:
    _, plan = _plan(cfg)
    overruns = [t for t in plan.truth if t.kind == "maintenance_overrun"]
    assert overruns
    for overrun in overruns:
        same_window = [
            t
            for t in plan.truth
            if t.kind == "maintenance_window" and t.org == overrun.org and t.start == overrun.start
        ]
        assert same_window == []
        assert overrun.end - overrun.start >= cfg.scenarios.overrun_minutes.low * MINUTE


def test_unplaceable_schedule_fails_loudly(tmp_path: Path) -> None:
    # No bursty keys anywhere: no steady key shares an org with one.
    path = variant(
        tmp_path,
        {
            'keys = ["bursty", "steady", "steady", "overdispersed"]': 'keys = ["steady", "steady", "overdispersed"]',
            'keys = ["bursty", "steady", "steady"]': 'keys = ["steady", "steady"]',
            "weight = 0.2\nrate_per_second = [2.0, 10.0]": "weight = 0.0\nrate_per_second = [2.0, 10.0]",
        },
    )
    cfg = load(path)
    with pytest.raises(PlanError, match="burst_on_steady_key: no eligible targets"):
        _plan(cfg)


def test_too_many_scenarios_for_the_world_fails_loudly(tmp_path: Path) -> None:
    # Eight random orgs can host at most eight onboarding cases.
    cfg = load(variant(tmp_path, {"onboarding = 1": "onboarding = 9"}))
    with pytest.raises(PlanError, match="onboarding: placed 8 of 9"):
        _plan(cfg)
