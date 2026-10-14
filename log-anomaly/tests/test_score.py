from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from logsim.output import iso
from logsim.plan import MINUTE
from logsim.score import Case, Page, Run, ScoreError, load_pages, load_run, parse_time, score, wilson

DEADLINE = 15.0


def _page(case: Case, at: int, run: Run) -> Page:
    if case.key is not None:
        return Page(0, at, run.key_org[case.key], case.key)
    return Page(0, at, case.org, None)


def _countable(run: Run) -> tuple[list[Case], list[Case]]:
    report = score(run, [], DEADLINE)
    return report.must_alert, report.must_stay_quiet


def test_parse_time_keeps_nanoseconds_and_offsets() -> None:
    assert parse_time("2026-10-19T00:00:00.000000001Z") == 1_792_368_000_000_000_001
    assert parse_time("2026-10-19T02:00:00.5+02:00") == 1_792_368_000_500_000_000
    with pytest.raises(ScoreError):
        parse_time("2026-10-19T00:00:00")


def test_wilson_interval() -> None:
    low, high = wilson(18, 20)
    assert round(low, 2) == 0.70 and round(high, 2) == 0.97


def test_an_oracle_scores_perfectly(run_dir: Path) -> None:
    run = load_run(run_dir)
    must, _ = _countable(run)
    assert must
    report = score(run, [_page(c, c.start, run) for c in must], DEADLINE).as_dict()
    assert report["recall"]["rate"] == 1.0
    assert report["pages"] == {"true": len(must)}
    assert report["precision"] == 1.0


def test_repeat_pages_are_duplicates_not_false(run_dir: Path) -> None:
    run = load_run(run_dir)
    must, _ = _countable(run)
    pages = [_page(c, c.start + i * MINUTE, run) for c in must for i in range(2)]
    assert score(run, pages, DEADLINE).as_dict()["pages"] == {"duplicate": len(must), "true": len(must)}


def test_silence_detects_nothing_and_pages_nothing(run_dir: Path) -> None:
    report = score(load_run(run_dir), [], DEADLINE).as_dict()
    assert report["recall"]["rate"] == 0.0
    assert report["pages"] == {} and report["precision"] is None


def test_paging_on_every_quiet_case_is_all_false(run_dir: Path) -> None:
    run = load_run(run_dir)
    _, quiet = _countable(run)
    pages = [_page(c, c.start, run) for c in quiet if c.kind != "chance_error_clump"]
    pages += [Page(0, c.start, None, None) for c in quiet if c.kind == "chance_error_clump"]
    report = score(run, pages, DEADLINE)
    assert report.pages == {"false": len(pages)}
    assert report.paged_quiet == {c.id for c in quiet}
    assert "unexplained" not in report.false_by_cause and "unattributed" not in report.false_by_cause


def test_the_wrong_key_does_not_detect(run_dir: Path) -> None:
    run = load_run(run_dir)
    must, _ = _countable(run)
    case = next(c for c in must if c.kind == "burst_on_steady_key")
    other = next(k for k, o in run.key_org.items() if o == run.key_org[case.key] and k != case.key)
    report = score(run, [Page(0, case.start, run.key_org[other], other)], DEADLINE)
    assert case.id not in report.detected and report.pages["false"] == 1


def test_naming_the_org_is_not_enough_for_a_key_case(run_dir: Path) -> None:
    run = load_run(run_dir)
    must, _ = _countable(run)
    case = next(c for c in must if c.kind == "key_silence")
    report = score(run, [Page(0, case.start, run.key_org[case.key], None)], DEADLINE)
    assert case.id not in report.detected


def test_the_deadline_is_inclusive_and_exact(run_dir: Path) -> None:
    run = load_run(run_dir)
    must, _ = _countable(run)
    case = must[0]
    last = case.end + int(DEADLINE * MINUTE)
    assert case.id in score(run, [_page(case, last, run)], DEADLINE).detected
    assert case.id not in score(run, [_page(case, last + 1, run)], DEADLINE).detected


def test_unknown_or_mismatched_ids_are_rejected(run_dir: Path, tmp_path: Path) -> None:
    run = load_run(run_dir)
    key, org = next(iter(run.key_org.items()))
    stranger = next(o for o in run.orgs if o != org)
    for doc, message in [
        ({"time": "2026-10-25T00:00:00Z", "api_key.id": "key_nope"}, "unknown api_key.id"),
        ({"time": "2026-10-25T00:00:00Z", "org.id": "org_nope"}, "unknown org.id"),
        ({"time": "2026-10-25T00:00:00Z", "api_key.id": key, "org.id": stranger}, "belongs to"),
        ({"time": "yesterday"}, "RFC 3339"),
    ]:
        path = tmp_path / "pages.jsonl"
        path.write_text(json.dumps(doc) + "\n")
        with pytest.raises(ScoreError, match=message):
            load_pages(path, run)


def _streamed(run_dir: Path, tmp_path: Path, events: list[dict[str, object]]) -> Path:
    copy = tmp_path / "streamed"
    copy.mkdir()
    for name in ("truth.jsonl", "world.json"):
        shutil.copy(run_dir / name, copy / name)
    manifest = json.loads((run_dir / "manifest.json").read_text())
    for field in ("hours_generated", "records", "logs_sha256"):
        del manifest[field]
    (copy / "manifest.json").write_text(json.dumps(manifest))
    (copy / "journal.jsonl").write_text("".join(json.dumps(e) + "\n" for e in events))
    return copy


def test_stream_gaps_exclude_cases_and_pages(run_dir: Path, tmp_path: Path) -> None:
    whole = load_run(run_dir)
    must, _ = _countable(whole)
    case = must[0]
    gap = (case.start - 5 * MINUTE, case.start + MINUTE)
    manifest = json.loads((run_dir / "manifest.json").read_text())
    copy = _streamed(
        run_dir,
        tmp_path,
        [
            {"event": "emitting", "emitting_from": manifest["start"]},
            {"event": "stopped", "sent_through": iso(gap[0])},
            {"event": "emitting", "emitting_from": iso(gap[1])},
            {"event": "finished"},
        ],
    )
    run = load_run(copy)
    assert run.gaps == (gap,)
    report = score(run, [_page(case, case.start, run)], DEADLINE)
    assert case.id not in {c.id for c in report.must_alert}
    assert report.pages == {"excluded": 1}


def test_an_unfinished_stream_needs_consent(run_dir: Path, tmp_path: Path) -> None:
    manifest = json.loads((run_dir / "manifest.json").read_text())
    through = parse_time(manifest["scored_from"]) + 600 * MINUTE
    copy = _streamed(
        run_dir,
        tmp_path,
        [
            {"event": "emitting", "emitting_from": manifest["start"]},
            {"event": "progress", "sent_through": iso(through)},
        ],
    )
    with pytest.raises(ScoreError, match="allow-unfinished"):
        load_run(copy)
    run = load_run(copy, allow_unfinished=True)
    assert run.horizon == through
    assert all(c.end + DEADLINE * MINUTE < through for c in score(run, [], DEADLINE).must_alert)


def test_warm_up_pages_are_not_scored(run_dir: Path) -> None:
    run = load_run(run_dir)
    warm = next(c for c in run.cases if not c.scored and c.kind == "maintenance_window")
    report = score(run, [_page(warm, warm.start, run), Page(0, run.scored_from - 1, None, None)], DEADLINE)
    assert report.pages == {"excluded": 2}
