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
from logsim.plan import HOUR, MINUTE, NS, Burst, ErrorSpan, Plan, build_plan
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
    assert digest.hexdigest() == "5ceec71f80741ec63eb8fa58e9a7f6d91160af940abaa50e401fa7325b1aa436"


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
    outage_kinds = ("maintenance_window", "maintenance_overrun", "unplanned_outage", "provider_outage")
    outages: dict[str, list[tuple[int, int]]] = {}
    for t in plan.truth:
        if t.kind in outage_kinds:
            for org in t.detail["orgs"] if t.org is None else [world.orgs[t.org].id]:
                outages.setdefault(org, []).append((t.start, t.end))
    assert any(t.kind == "provider_outage" for t in plan.truth)
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
                "retry_storm = 2": "retry_storm = 0",
                "error_rate_shift = 2": "error_rate_shift = 0",
                "provider_outage = 1": "provider_outage = 0",
                "bad_deploy = 1": "bad_deploy = 0",
                "renamed_error = 1": "renamed_error = 0",
                "latency_leak = 1": "latency_leak = 0",
                "organic_growth = 1": "organic_growth = 0",
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


def _hours(cfg: Config, world: World, plan: Plan, start: int, end: int) -> list[Record]:
    first, last = (start - plan.start) // HOUR, (end - 1 - plan.start) // HOUR
    return [r for i in range(first, min(last, plan.chunks - 1) + 1) for r in generate_chunk(cfg, world, plan, SEED, i)]


def _status(records: list[Record]) -> Counter[int]:
    return Counter(r.attributes["http.response.status_code"] for r in records)


def _within(observed: int, expected: float) -> bool:
    return abs(observed - expected) <= 5 * np.sqrt(expected) + 2


def test_a_shift_raises_errors_and_leaves_volume_alone(cfg: Config, small_run: Run) -> None:
    world, plan, _ = small_run
    shifts = [t for t in plan.truth if t.kind == "error_rate_shift"]
    assert shifts
    for truth in shifts:
        key = world.keys[truth.key]
        spans = {
            k: tuple(s for s in v if (s.start, s.end) != (truth.start, truth.end)) for k, v in plan.error_spans.items()
        }
        without = replace(plan, error_spans=spans)
        shifted = _requests(_hours(cfg, world, plan, truth.start, truth.end), key.id, truth.start, truth.end)
        plain = _requests(_hours(cfg, world, without, truth.start, truth.end), key.id, truth.start, truth.end)
        # Same requests, to the nanosecond; only their outcomes moved.
        assert [r.time for r in shifted] == [r.time for r in plain] and shifted
        assert _within(_status(shifted)[500], len(shifted) * truth.detail["server_error_share"])
        assert _status(shifted)[500] > _status(plain)[500]


def test_a_batch_raises_volume_and_leaves_the_error_share_alone(cfg: Config, small_run: Run) -> None:
    world, plan, _ = small_run
    surges = [t for t in plan.truth if t.kind == "batch_surge" and t.scored][:4]
    assert surges
    for truth in surges:
        key = world.keys[truth.key]
        without = replace(plan, surges={k: () for k in plan.surges})
        batch = _requests(_hours(cfg, world, plan, truth.start, truth.end), key.id, truth.start, truth.end)
        plain = _requests(_hours(cfg, world, without, truth.start, truth.end), key.id, truth.start, truth.end)
        assert plain and _within(len(batch), len(plain) * truth.detail["multiplier"])
        assert _within(_status(batch)[500], len(batch) * cfg.traffic.server_error_share)


def _cascade_case(plan: Plan, kind: str) -> tuple[object, object]:
    truth = next(t for t in plan.truth if t.kind == kind and t.scored)
    (cascade,) = [c for c in plan.cascades if (c.key, c.start) == (truth.key, truth.start)]
    return truth, cascade


@pytest.mark.parametrize("kind", ["retry_storm", "retry_echo"])
def test_retries_fail_while_overloaded_and_succeed_after(cfg: Config, small_run: Run, kind: str) -> None:
    """Every attempt in the cascade is emitted: during the overload every request
    the key sends, original or retry, fails with 503; afterwards none does."""
    world, plan, _ = small_run
    truth, cascade = _cascade_case(plan, kind)
    key = world.keys[truth.key]
    others = tuple(c for c in plan.cascades if c is not cascade)
    spans = {k: tuple(s for s in v if s.start != cascade.start) for k, v in plan.error_spans.items()}
    without = replace(plan, cascades=others, error_spans=spans)
    end = truth.end + 10 * MINUTE
    emitted = _requests(_hours(cfg, world, plan, truth.start, end), key.id, truth.start, end)
    plain = _requests(_hours(cfg, world, without, truth.start, end), key.id, truth.start, end)
    assert len(emitted) == len(plain) + sum(cascade.attempts)
    during = [r for r in emitted if r.time < cascade.overload_end]
    after = [r for r in emitted if r.time >= cascade.overload_end]
    assert during and set(_status(during)) == {503}
    assert 503 not in _status(after)


def test_a_leak_slows_its_org_in_proportion_and_stops_at_restart(cfg: Config, small_run: Run) -> None:
    world, plan, _ = small_run
    (truth,) = (t for t in plan.truth if t.kind == "latency_leak")
    org = world.orgs[truth.org]
    without = replace(plan, latency_ramps={})
    late = truth.end - 2 * HOUR, truth.end + HOUR

    def latencies(p: Plan) -> list[tuple[int, int]]:
        records = _hours(cfg, world, p, *late)
        return [
            (r.time, r.attributes["duration_ms"])
            for r in records
            if r.attributes.get("org.id") == org.id and "http.route" in r.attributes
        ]

    leaking, plain = latencies(plan), latencies(without)
    assert [t for t, _ in leaking] == [t for t, _ in plain] and leaking
    for (t, ms), (_, base) in zip(leaking, plain, strict=True):
        if t >= truth.end:
            assert ms == base
        else:
            factor = 1 + (truth.detail["factor"] - 1) * (t - truth.start) / (truth.end - truth.start)
            # Both sides are rounded to whole milliseconds.
            assert abs(ms - base * factor) <= factor / 2 + 0.5, (ms, base, factor)


def test_growth_raises_an_orgs_volume_and_keeps_it(cfg: Config, small_run: Run) -> None:
    world, plan, _ = small_run
    (truth,) = (t for t in plan.truth if t.kind == "organic_growth")
    org = world.orgs[truth.org]
    without = replace(plan, rate_ramps={})
    after = truth.end, plan.end

    def volume(p: Plan) -> int:
        return sum(
            1
            for r in _hours(cfg, world, p, *after)
            if r.attributes.get("org.id") == org.id and "http.route" in r.attributes and r.time >= truth.end
        )

    grown, plain = volume(plan), volume(without)
    assert plain and _within(grown, plain * truth.detail["factor"])


def test_a_bad_deploy_fails_its_providers_orgs_and_no_one_else(cfg: Config, small_run: Run) -> None:
    world, plan, _ = small_run
    (truth,) = (t for t in plan.truth if t.kind == "bad_deploy")
    members = set(truth.detail["orgs"])
    records = [r for r in _hours(cfg, world, plan, truth.start, truth.end) if truth.start <= r.time < truth.end]
    hit = [r for r in records if "http.route" in r.attributes and r.attributes["org.id"] in members]
    spared = [r for r in records if "http.route" in r.attributes and r.attributes["org.id"] not in members]
    assert hit and spared
    assert _within(_status(hit)[500], len(hit) * truth.detail["server_error_share"])
    assert _within(_status(spared)[500], len(spared) * cfg.traffic.server_error_share)
    deploys = [r for r in records if r.service == "deployer"]
    assert deploys[0].time == truth.start and deploys[0].body == f"deployed release {truth.detail['release']}"


def test_deploys_are_logged_once_each(small_run: Run) -> None:
    _, plan, records = small_run
    logged = [(r.time, r.attributes["deploy.release"]) for r in records if r.service == "deployer"]
    assert logged == [(d.time, d.release) for d in plan.deploys] and logged


def test_a_renamed_message_switches_text_at_its_deploy(cfg: Config, small_run: Run) -> None:
    _, plan, records = small_run
    (rename,) = plan.renames
    old = cfg.background_errors[rename.template].template.text
    patterns = {
        text: re.compile(re.sub(r"\\\{(ms|id|n)\\\}", ".+", re.escape(text)) + "$") for text in (old, rename.to.text)
    }
    before = [r for r in records if r.time < rename.time]
    after = [r for r in records if r.time >= rename.time]
    assert [r for r in before if patterns[old].match(r.body)]
    assert not [r for r in after if patterns[old].match(r.body)]
    assert not [r for r in before if patterns[rename.to.text].match(r.body)]
    assert [r for r in after if patterns[rename.to.text].match(r.body)]


@pytest.mark.parametrize("overload_first", [True, False])
def test_overlapping_failures_take_the_worse(cfg: Config, small_run: Run, overload_first: bool) -> None:
    """An overloaded dependency fails every request, whatever else is failing some of them."""
    world, plan, _ = small_run
    key = next(k for k in world.keys if k.org not in plan.activations and k.org not in plan.rate_ramps)
    start = plan.scored_from + 3 * HOUR
    overload = ErrorSpan(key.index, start, start + HOUR, 1.0, 503)
    partial = ErrorSpan(key.index, start, start + HOUR, 0.1, 500)
    spans = (overload, partial) if overload_first else (partial, overload)
    planted = replace(plan, error_spans={**plan.error_spans, key.index: spans})
    requests = _requests(_hours(cfg, world, planted, start, start + HOUR), key.id, start, start + HOUR)
    assert requests and set(_status(requests)) == {503}
