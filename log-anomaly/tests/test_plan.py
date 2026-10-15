from __future__ import annotations

from collections import Counter
from datetime import UTC, datetime
from itertools import pairwise
from pathlib import Path

import pytest
from conftest import SEED, START, variant

from logsim.config import Config, load
from logsim.kinds import SCOPE, SIGNALS
from logsim.plan import DAY, HOUR, MINUTE, NS, Activation, Plan, PlanError, Truth, _Builder, _wall, build_plan
from logsim.repro import zone
from logsim.world import World, build_world

NATURAL = ("natural_burst", "maintenance_window", "scheduled_export", "batch_surge", "retry_echo")


def _plan(cfg: Config, seed: int = SEED) -> tuple[World, Plan]:
    world = build_world(cfg, seed)
    return world, build_plan(cfg, world, seed, START)


def _orgs(world: World, truth: Truth) -> set[int]:
    if truth.org is not None:
        return {truth.org}
    ids = set(truth.detail["orgs"])
    return {o.index for o in world.orgs if o.id in ids}


def test_plan_is_a_pure_function_of_config_seed_and_start(cfg: Config) -> None:
    assert _plan(cfg)[1] == _plan(cfg)[1]
    assert _plan(cfg)[1].truth != _plan(cfg, seed=SEED + 1)[1].truth


def test_start_must_fall_on_a_utc_hour(cfg: Config) -> None:
    world = build_world(cfg, SEED)
    with pytest.raises(PlanError, match="UTC hour"):
        build_plan(cfg, world, SEED, START + 30 * MINUTE)


def test_every_injected_case_sits_inside_the_scored_phase(cfg: Config) -> None:
    _, plan = _plan(cfg)
    injected = [t for t in plan.truth if t.kind not in NATURAL]
    assert len(injected) == 21  # the small config's scenario counts, renamed_error aside
    for truth in injected:
        assert plan.scored_from <= truth.start < truth.end <= plan.end, truth


def test_alerts_have_matched_quiet_cases(cfg: Config) -> None:
    _, plan = _plan(cfg)
    by_pair = Counter((t.pair, t.expect) for t in plan.truth if t.scored)
    for pair in ("burst", "outage", "egress", "retry", "error_rate", "drift"):
        assert by_pair[(pair, "alert")] > 0 and by_pair[(pair, "quiet")] > 0, pair


def test_no_target_repeats_within_a_kind_while_others_remain(cfg: Config) -> None:
    _, plan = _plan(cfg)
    targets = Counter(
        (t.kind, t.key if t.key is not None else t.org if t.org is not None else t.detail["provider"])
        for t in plan.truth
        if t.expect == "alert"
    )
    assert max(targets.values()) == 1


def test_bad_deploys_spread_over_providers(tmp_path: Path) -> None:
    """The small run has one bad deploy, so the check above cannot see a repeat."""
    cfg = load(variant(tmp_path, {"bad_deploy = 1": "bad_deploy = 3"}))
    _, plan = _plan(cfg)
    providers = [t.detail["provider"] for t in plan.truth if t.kind == "bad_deploy"]
    assert len(providers) == 3 and len(set(providers)) == 3


def test_scenarios_in_one_org_keep_their_distance(cfg: Config) -> None:
    """Dedicated orgs aside: a provider-wide failure takes them down as
    bystanders whenever it happens."""
    world, plan = _plan(cfg)
    gap = int(cfg.scenarios.min_gap_hours * HOUR)
    dedicated = set(plan.activations) | set(plan.rate_ramps)
    by_org = sorted(
        (org, t.start, t.end, t.kind)
        for t in plan.truth
        if t.kind not in NATURAL
        for org in _orgs(world, t)
        if not (org in dedicated and t.kind in ("provider_outage", "bad_deploy"))
    )
    assert any(kind == "provider_outage" for *_, kind in by_org)
    for a, b in pairwise(by_org):
        if a[0] == b[0]:
            assert b[1] >= a[2] + gap, (a, b)


def test_dedicated_orgs_are_excluded_from_other_scenarios(cfg: Config) -> None:
    """Onboarding and organic growth each own their org: no other case targets
    it. A provider-wide failure still takes it down with its provider."""
    world, plan = _plan(cfg)
    (newcomer,) = plan.activations
    (grower,) = plan.rate_ramps
    provider_wide = {"provider_outage", "bad_deploy"}
    for org in (newcomer, grower):
        assert {t.kind for t in plan.truth if t.expect == "alert" and org in _orgs(world, t)} <= provider_wide
        own = [t for t in plan.truth if t.org == org and t.kind not in NATURAL]
        assert [t.kind for t in own] == ["onboarding" if org == newcomer else "organic_growth"]
        spans = {(s.start, s.end) for k in world.orgs[org].keys for s in plan.error_spans.get(k, ())}
        echoes = {(c.start, c.overload_end) for c in plan.cascades if world.keys[c.key].org == org}
        bad = {(t.start, t.end) for t in plan.truth if t.kind == "bad_deploy" and org in _orgs(world, t)}
        assert spans <= echoes | bad
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


def test_provider_outage_takes_down_every_org_on_one_provider(cfg: Config) -> None:
    world, plan = _plan(cfg)
    for truth in (t for t in plan.truth if t.kind == "provider_outage"):
        members = _orgs(world, truth)
        expected = {
            o.index
            for o in world.orgs
            if o.provider.name == truth.detail["provider"] and plan.active(o.index, truth.end - 1)
        }
        assert members == expected and len(members) >= 2
        for org in members:
            assert any(o.start == truth.start and o.end == truth.end for o in plan.outages[org])


def test_a_bad_deploy_starts_at_a_deploy_and_ends_at_its_fix(cfg: Config) -> None:
    world, plan = _plan(cfg)
    deploys = {d.time: d.release for d in plan.deploys}
    (truth,) = (t for t in plan.truth if t.kind == "bad_deploy")
    assert deploys[truth.start] == truth.detail["release"] and deploys[truth.end] == truth.detail["fix_release"]
    keys = {k for o in _orgs(world, truth) for k in world.orgs[o].keys}
    spans = {k for k, v in plan.error_spans.items() for s in v if (s.start, s.end) == (truth.start, truth.end)}
    assert spans == keys
    # Releases only count up: the fix is a forward deploy, not a rollback.
    assert [d.release for d in plan.deploys] == sorted(d.release for d in plan.deploys)


def test_renames_happen_at_scored_deploys_no_other_scenario_claimed(cfg: Config) -> None:
    _, plan = _plan(cfg)
    deploys = {d.time for d in plan.deploys}
    bad = {t.start for t in plan.truth if t.kind == "bad_deploy"}
    assert plan.renames
    for rename in plan.renames:
        assert rename.time in deploys and rename.time >= plan.scored_from and rename.time not in bad


def test_storms_reach_the_admission_limit_and_echoes_die_out(cfg: Config) -> None:
    _, plan = _plan(cfg)
    cap = cfg.retries.storm_max_rate_per_second * cfg.traffic.rate_scale
    storms = [t for t in plan.truth if t.kind == "retry_storm"]
    echoes = [t for t in plan.truth if t.kind == "retry_echo"]
    assert storms and echoes
    by_start = {(c.key, c.start): c for c in plan.cascades}
    for storm in storms:
        cascade = by_start[(storm.key, storm.start)]
        assert max(cascade.attempts) >= cap and cascade.overload_end == storm.end
        # The circuit breaker opens when the overload ends: no retry after it.
        assert cascade.start + len(cascade.attempts) * NS <= storm.end
    smallest_storm = min(sum(by_start[(s.key, s.start)].attempts) for s in storms)
    for echo in echoes:
        cascade = by_start[(echo.key, echo.start)]
        # The dependency recovers after the blip; retries trail off and the case ends with the last one.
        assert cascade.overload_end - cascade.start <= cfg.retries.blip_seconds.high * NS
        assert echo.end == max(cascade.overload_end, cascade.start + len(cascade.attempts) * NS)
        assert sum(cascade.attempts) < smallest_storm / 10


def test_alert_cases_keep_clear_of_quiet_look_alikes(cfg: Config) -> None:
    """No page can be both right about an alert case and charged to a natural
    quiet case: they never overlap on a shared entity and a shared signal."""
    world, plan = _plan(cfg)
    margin = 30 * MINUTE
    alerts = [t for t in plan.truth if t.expect == "alert"]
    quiet = [t for t in plan.truth if t.kind in NATURAL]
    checked = 0
    for a in alerts:
        for q in quiet:
            if not SIGNALS[a.kind] & SIGNALS[q.kind]:
                continue
            if q.end <= a.start - margin or q.start >= a.end + margin:
                continue
            if a.kind == "maintenance_overrun" and q.start == a.start:
                continue
            checked += 1
            if SCOPE[a.kind] == "key":
                assert q.key != a.key and not (q.key is None and q.org == world.keys[a.key].org), (a, q)
            else:
                assert q.org not in _orgs(world, a), (a, q)
    assert checked, "nothing overlapped in time; the test exercised nothing"


def test_deploys_ship_on_weekdays_in_working_hours(cfg: Config) -> None:
    """Fixes included: one shipped at night would say the last release broke."""
    _, plan = _plan(cfg)
    tz = zone(cfg.deploys.timezone)
    assert any(t.kind == "bad_deploy" for t in plan.truth)
    for deploy in plan.deploys:
        local = datetime.fromtimestamp(deploy.time // NS, tz)
        assert local.weekday() < 5 and cfg.deploys.local_hour.low <= local.hour < cfg.deploys.local_hour.high, local


@pytest.mark.parametrize(("hour", "placed"), [(16.75, False), (15.0, True)])
def test_a_fix_deploy_waits_for_working_hours(cfg: Config, hour: float, placed: bool) -> None:
    """Direct, because few seeds draw a bad deploy late enough: from 16:45 the
    fix (20-60 minutes later) would ship after 17:00, so it is never placed."""
    world = build_world(cfg, SEED)
    monday = int(datetime(2026, 10, 19, tzinfo=UTC).timestamp()) * NS
    builder = _Builder(cfg, world, SEED, monday)
    tz = zone(cfg.deploys.timezone)
    day = datetime.fromtimestamp((builder.scored_from + DAY) // NS, tz).date()
    assert day.weekday() < 5
    # The run goes on well past that evening, so nothing else rules the fix out.
    assert _wall(day, 20 * 3600, tz) < builder.end
    builder.natural_deploys = [_wall(day, int(hour * 3600), tz)]
    if placed:
        builder._bad_deploys()
        assert all(builder._in_deploy_hours(t) for t in builder.fix_deploys) and builder.fix_deploys
    else:
        with pytest.raises(PlanError, match="bad_deploy: placed 0 of"):
            builder._bad_deploys()


def test_clashes_follow_the_scorers_matching(cfg: Config) -> None:
    """Direct, because a seed rarely puts a case next to a look-alike: an org-wide
    case clashes with anything in its orgs, a key case only with its key, and
    neither with a quiet case showing other signals."""
    world = build_world(cfg, SEED)
    builder = _Builder(cfg, world, SEED, START)
    key = world.keys[0]
    other = next(k for k in world.keys if k.org != key.org)
    hour = builder.scored_from + HOUR
    builder.quiet = [("batch_surge", key.index, key.org, hour, hour + HOUR)]
    assert builder._clashes("bad_deploy", (key.org,), None, hour, hour + HOUR)
    assert not builder._clashes("bad_deploy", (other.org,), None, hour, hour + HOUR)
    assert builder._clashes("error_rate_shift", (key.org,), key.index, hour, hour + HOUR)
    assert not builder._clashes("error_rate_shift", (other.org,), other.index, hour, hour + HOUR)
    assert not builder._clashes("latency_leak", (key.org,), None, hour, hour + HOUR)
    # Starting within 30 minutes of its end counts; starting 30 minutes after does not.
    edge = hour + HOUR + 30 * MINUTE
    assert builder._clashes("bad_deploy", (key.org,), None, edge - 1, edge + HOUR)
    assert not builder._clashes("bad_deploy", (key.org,), None, edge, edge + HOUR)


@pytest.mark.parametrize(
    ("weekday", "hour", "inside"),
    [(0, 9.0, True), (4, 16.99, True), (0, 8.99, False), (0, 17.0, False), (5, 12.0, False)],
)
def test_deploy_hours_are_weekday_working_hours(cfg: Config, weekday: int, hour: float, inside: bool) -> None:
    world = build_world(cfg, SEED)
    builder = _Builder(cfg, world, SEED, START)
    tz = zone(cfg.deploys.timezone)
    day = datetime.fromtimestamp(builder.scored_from // NS, tz).date()
    day = day.fromordinal(day.toordinal() + (weekday - day.weekday()) % 7)
    assert builder._in_deploy_hours(_wall(day, int(hour * 3600), tz)) is inside


def test_a_bad_deploy_needs_an_org_that_exists(cfg: Config) -> None:
    world = build_world(cfg, SEED)
    builder = _Builder(cfg, world, SEED, START)
    builder._deploys()
    for org in world.orgs:
        builder.activations[org.index] = Activation(builder.end, builder.end + DAY)
    with pytest.raises(PlanError, match="bad_deploy: placed 0 of"):
        builder._bad_deploys()


def test_dedicated_orgs_fail_with_their_provider_as_bystanders(cfg: Config) -> None:
    """Direct, because a seed may never put a dedicated org on the failing
    provider. Sparing it would mark it as a quiet case; it fails with the rest,
    and placement neither needs room in it nor books any."""
    world = build_world(cfg, SEED)
    builder = _Builder(cfg, world, SEED, START)
    name, orgs = builder._outage_providers()[0]
    bystander = orgs[0]
    builder.dedicated.add(bystander.index)
    builder.occupied[bystander.index].append((builder.scored_from, builder.end))
    assert bystander in builder._provider_orgs()[name]
    builder._provider_outages()
    (outage,) = builder.injected
    assert bystander.id in outage.detail["orgs"] and builder.outages[bystander.index]
    assert builder.occupied[bystander.index] == [(builder.scored_from, builder.end)]


def test_an_org_still_to_arrive_is_no_member(cfg: Config) -> None:
    world = build_world(cfg, SEED)
    builder = _Builder(cfg, world, SEED, START)
    _, orgs = builder._outage_providers()[0]
    hour = builder.scored_from + DAY
    builder.activations[orgs[0].index] = Activation(hour + HOUR, hour + DAY)
    assert orgs[0] not in builder._members(orgs, hour + HOUR)
    assert orgs[0] in builder._members(orgs, hour + HOUR + 1)


def test_a_provider_outage_needs_two_orgs(cfg: Config) -> None:
    """With one org per provider existing for the whole run, none can fail."""
    world = build_world(cfg, SEED)
    builder = _Builder(cfg, world, SEED, START)
    for _, orgs in builder._outage_providers():
        for org in orgs[1:]:
            builder.activations[org.index] = Activation(builder.end, builder.end + DAY)
    with pytest.raises(PlanError, match="provider_outage: placed 0 of"):
        builder._provider_outages()


def test_a_storm_that_dies_out_is_never_placed(cfg: Config) -> None:
    """A supercritical cascade can still die in its first seconds. Forced to
    every time, placement must fail loudly rather than label an echo a storm."""
    world = build_world(cfg, SEED)
    builder = _Builder(cfg, world, SEED, START)
    builder._cascade = lambda *args, **kwargs: (1, 0, 0)
    with pytest.raises(PlanError, match="retry_storm: placed 0 of"):
        builder._retry_storms()


@pytest.mark.parametrize("kind", ["bad_deploy", "renamed_error"])
def test_a_claimed_deploy_is_not_claimed_twice(cfg: Config, kind: str) -> None:
    """With the only scored deploy already claimed, placement must fail loudly."""
    world = build_world(cfg, SEED)
    builder = _Builder(cfg, world, SEED, START)
    only = builder.scored_from + DAY + 10 * HOUR
    builder.natural_deploys = [only]
    builder.deploys_used = {only}
    place = builder._bad_deploys if kind == "bad_deploy" else builder._renamed_errors
    with pytest.raises(PlanError, match=f"{kind}: placed 0 of"):
        place()


def test_a_growing_org_is_out_of_every_later_draw(cfg: Config) -> None:
    world = build_world(cfg, SEED)
    builder = _Builder(cfg, world, SEED, START)
    builder._organic_growth()
    (grower,) = builder.rate_ramps
    assert grower not in {o.index for o in builder._eligible_orgs()}
    assert grower not in {k.org for k in builder._eligible_keys(lambda k, o: True)}
