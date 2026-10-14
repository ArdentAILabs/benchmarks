from __future__ import annotations

from pathlib import Path

import pytest

from logsim.baseline import ThresholdBaseline, mask, run_baseline
from logsim.plan import MINUTE
from logsim.score import Page, load_run, parse_time, score


def test_mask_hides_numbers_and_ids() -> None:
    assert mask("slow query on branch 765026ca: 1078ms") == "slow query on branch #: #ms"
    assert mask("upstream timeout after 2000ms") == "upstream timeout after #ms"


def test_baseline_catches_the_loud_cases_and_pays_for_the_quiet_ones(run_dir: Path) -> None:
    run = load_run(run_dir)
    key_org = run.key_org
    pages = [
        Page(i, parse_time(p["time"]), p.get("org.id"), p.get("api_key.id"), p["signal"])
        for i, p in enumerate(run_baseline(run_dir), start=1)
    ]
    assert all(p.org is None or p.key is None or key_org[p.key] == p.org for p in pages)
    report = score(run, pages, 15.0)
    by_kind = report.as_dict()["recall_by_kind"]
    for kind in (
        "burst_on_steady_key",
        "unplanned_outage",
        "maintenance_overrun",
        "novel_error",
        "retry_storm",
        "provider_outage",
        "bad_deploy",
        "latency_leak",
    ):
        assert by_kind[kind]["detected"] == by_kind[kind]["cases"], kind
    # Two failed health checks look the same planned or not, a chance clump of
    # a known error looks like a novel one, and a retry echo starts like a
    # storm: a threshold pays for all three.
    for cause in ("maintenance_window", "chance_error_clump", "retry_echo"):
        assert report.false_by_cause[cause] > 0, cause


def test_a_minute_without_records_is_refused() -> None:
    baseline = ThresholdBaseline(0, 60 * MINUTE)
    record = {"severity": "INFO", "service": "source-monitor", "body": "ok", "attributes": {"org.id": "org_a"}}
    baseline.feed({**record, "time_unix_nano": 0})
    with pytest.raises(ValueError, match="needs one a minute"):
        baseline.feed({**record, "time_unix_nano": 2 * MINUTE})
