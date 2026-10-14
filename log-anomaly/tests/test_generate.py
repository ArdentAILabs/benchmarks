from __future__ import annotations

import hashlib
import re
from collections import Counter
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from conftest import SEED, START, run, variant

from logsim.config import Config, load
from logsim.generate import Record, generate_chunk
from logsim.output import record_line
from logsim.plan import MINUTE, NS, Burst, Plan, build_plan
from logsim.world import World, build_world

Run = tuple[World, Plan, list[Record]]


def _requests(records: list[Record], key_id: str, start: int, end: int) -> list[Record]:
    return [
        r
        for r in records
        if r.attributes.get("api_key.id") == key_id and "http.route" in r.attributes and start <= r.time < end
    ]


def test_an_hour_is_the_same_alone_or_in_sequence(cfg: Config, small_run: Run) -> None:
    world, plan, records = small_run
    late = generate_chunk(cfg, world, plan, SEED, 40)
    early = generate_chunk(cfg, world, plan, SEED, 3)
    hour = [r for r in records if plan.start + 40 * 3600 * NS <= r.time < plan.start + 41 * 3600 * NS]
    assert late == hour
    assert early == [r for r in records if plan.start + 3 * 3600 * NS <= r.time < plan.start + 4 * 3600 * NS]


def test_output_is_pinned(cfg: Config, small_run: Run) -> None:
    """Fails when anything changes the bytes a seed produces: a code change, or a
    numpy or tzdata upgrade. Update the digest only for an intended change, and
    say in the commit that every earlier run's output is now different."""
    world, plan, _ = small_run
    digest = hashlib.sha256()
    for index in range(3):
        for record in generate_chunk(cfg, world, plan, SEED, index):
            digest.update((record_line(record) + "\n").encode())
    assert digest.hexdigest() == "95b9d89926c3f9d518e94ac0a381138932b6db3432d565fdffbe78d50d2d7fcc"


def test_records_are_time_ordered_and_inside_the_run(small_run: Run) -> None:
    _, plan, records = small_run
    times = [r.time for r in records]
    assert times == sorted(times)
    assert plan.start <= times[0] and times[-1] < plan.end


def test_injected_bursts_reach_their_rate(small_run: Run) -> None:
    world, plan, records = small_run
    for truth in (t for t in plan.truth if t.kind == "burst_on_steady_key"):
        key = world.keys[truth.key]
        seconds = (truth.end - truth.start) / NS
        expected = truth.detail["rate_per_second"] * seconds
        observed = len(_requests(records, key.id, truth.start, truth.end))
        assert abs(observed - expected) < 5 * np.sqrt(expected) + 2, (observed, expected)


def test_silenced_keys_emit_nothing_in_their_window(small_run: Run) -> None:
    """No requests, and no errors or exports either: a key logging errors while
    sending nothing would give the silence away."""
    world, plan, records = small_run
    for truth in (t for t in plan.truth if t.kind == "key_silence"):
        key_id = world.keys[truth.key].id
        assert [
            r for r in records if r.attributes.get("api_key.id") == key_id and truth.start <= r.time < truth.end
        ] == []
        assert _requests(records, key_id, truth.start - 3600 * NS, truth.start)


def test_health_fails_inside_outages_and_only_there(small_run: Run) -> None:
    world, plan, records = small_run
    outage_kinds = ("maintenance_window", "maintenance_overrun", "unplanned_outage")
    outages: dict[str, list[tuple[int, int]]] = {}
    for t in plan.truth:
        if t.kind in outage_kinds:
            outages.setdefault(world.orgs[t.org].id, []).append((t.start, t.end))
    checked = Counter()
    for r in records:
        if r.service != "source-monitor":
            continue
        down = any(s <= r.time < e for s, e in outages.get(r.attributes["org.id"], ()))
        assert (r.severity == "ERROR") == down, r
        checked[down] += 1
    assert checked[True] > 0 and checked[False] > 0


def test_novel_errors_appear_only_in_their_window(small_run: Run) -> None:
    world, plan, records = small_run
    for truth in (t for t in plan.truth if t.kind == "novel_error"):
        pattern = re.compile(re.sub(r"\\\{(ms|id|n)\\\}", ".+", re.escape(truth.detail["template"])) + "$")
        hits = [r for r in records if pattern.match(r.body)]
        assert len(hits) == truth.detail["count"]
        assert all(truth.start <= r.time <= truth.end for r in hits)
        assert {r.attributes["api_key.id"] for r in hits} == {world.keys[truth.key].id}


def test_bulk_exports_carry_their_bytes(small_run: Run) -> None:
    world, plan, records = small_run
    for truth in (t for t in plan.truth if t.kind in ("unscheduled_bulk_export", "scheduled_export")):
        key_id = world.keys[truth.key].id
        parts = [
            r
            for r in records
            if r.attributes.get("api_key.id") == key_id
            and "export.bytes" in r.attributes
            and truth.start <= r.time < truth.end
        ]
        if truth.end <= plan.end:
            assert sum(r.attributes["export.bytes"] for r in parts) >= truth.detail["bytes"]


def test_onboarding_org_is_silent_before_it_arrives(small_run: Run) -> None:
    world, plan, records = small_run
    (truth,) = (t for t in plan.truth if t.kind == "onboarding")
    org_id = world.orgs[truth.org].id
    assert not [r for r in records if r.attributes.get("org.id") == org_id and r.time < truth.start]
    assert [r for r in records if r.attributes.get("org.id") == org_id and r.time >= truth.end]


def _shapes(records: list[Record]) -> set[tuple[str, str, tuple[str, ...]]]:
    return {(r.service, r.severity, tuple(r.attributes)) for r in records}


def test_injection_leaves_no_structural_trace(tmp_path: Path, small_run: Run) -> None:
    """Records from injected cases must look like natural ones: same services,
    severities, attribute keys, and (for requests and health checks) bodies."""
    _, _, injected = small_run
    quiet = load(
        variant(
            tmp_path,
            {
                "burst_on_steady_key = 2": "burst_on_steady_key = 0",
                "unplanned_outage = 2": "unplanned_outage = 0",
                "maintenance_overrun = 2": "maintenance_overrun = 0",
                "novel_error = 2": "novel_error = 0",
                "unscheduled_bulk_export = 2": "unscheduled_bulk_export = 0",
                "key_silence = 2": "key_silence = 0",
                "onboarding = 1": "onboarding = 0",
            },
        )
    )
    _, _, natural = run(quiet)
    assert _shapes(injected) == _shapes(natural)

    def bodies(records: list[Record]) -> set[str]:
        return {r.body for r in records if r.service == "source-monitor" or "http.route" in r.attributes}

    assert bodies(injected) == bodies(natural)


@pytest.mark.parametrize(("archetype", "low", "high"), [("steady", 0.85, 1.15), ("overdispersed", 1.5, None)])
def test_count_dispersion_matches_the_archetype(tmp_path: Path, archetype: str, low: float, high: float | None) -> None:
    """Per-minute variance/mean: Poisson keys sit near 1, gamma-mixed keys well above."""
    flat = load(
        variant(
            tmp_path,
            {
                "diurnal_amplitude = 0.6": "diurnal_amplitude = 0.0",
                "weekend_factor = 0.4": "weekend_factor = 1.0",
                "rate_scale = 0.05": "rate_scale = 1.0",
            },
        )
    )
    world = build_world(flat, SEED)
    plan = build_plan(flat, world, SEED, START)
    keys = [k for k in world.keys if k.archetype == archetype and not plan.bursts.get(k.index)]
    hours = range(24, 30)  # scored day one, before any onboarding ramp matters
    records = [r for i in hours for r in generate_chunk(flat, world, plan, SEED, i)]
    ratios = []
    for key in keys:
        if key.org in plan.activations or any(s.key == key.index for v in plan.silences.values() for s in v):
            continue
        minutes = Counter(
            (r.time - plan.start) // MINUTE
            for r in records
            if r.attributes.get("api_key.id") == key.id and "http.route" in r.attributes
        )
        counts = np.array([minutes.get(m, 0) for m in range(24 * 60, 30 * 60)])
        ratios.append(counts.var() / counts.mean())
    assert ratios
    ratio = float(np.median(ratios))
    assert ratio >= low and (high is None or ratio <= high), ratio


def test_no_burst_before_an_org_arrives(cfg: Config, small_run: Run) -> None:
    world, plan, _ = small_run
    (org,) = plan.activations
    key = world.orgs[org].keys[0]
    arrival = plan.activations[org].start
    early = Burst(key, arrival - 2 * 3600 * NS, arrival - 3600 * NS, 1000.0)
    planted = replace(plan, bursts={**plan.bursts, key: (*plan.bursts.get(key, ()), early)})
    index = (early.start - plan.start) // (3600 * NS)
    assert not [
        r
        for r in generate_chunk(cfg, world, planted, SEED, index)
        if r.attributes.get("api_key.id") == world.keys[key].id
    ]


def test_silence_holds_under_heavy_errors_and_exports(tmp_path: Path) -> None:
    """At the shipped rates an error or export rarely lands on a silenced key,
    so this run makes both near-certain and checks none get through."""
    heavy = load(
        variant(
            tmp_path,
            {"rate_per_hour = 6.0": "rate_per_hour = 6000.0", "ad_hoc_per_day = 2.0": "ad_hoc_per_day = 200.0"},
        )
    )
    world = build_world(heavy, SEED)
    plan = build_plan(heavy, world, SEED, START)
    silences = [t for t in plan.truth if t.kind == "key_silence"]
    assert silences
    for truth in silences:
        key_id = world.keys[truth.key].id
        hours = range((truth.start - plan.start) // (3600 * NS), (truth.end - plan.start) // (3600 * NS) + 1)
        records = [r for i in hours for r in generate_chunk(heavy, world, plan, SEED, i)]
        assert [
            r for r in records if r.attributes.get("api_key.id") == key_id and truth.start <= r.time < truth.end
        ] == []
        assert [r for r in records if r.attributes.get("api_key.id") == key_id and "http.route" not in r.attributes]
