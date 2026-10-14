from __future__ import annotations

import json
from pathlib import Path

import pytest
from conftest import SEED, SMALL

from logsim.cli import main

START = "2026-10-23T00:00:00Z"


def test_generate_writes_a_complete_run(tmp_path: Path) -> None:
    out = tmp_path / "run"
    assert (
        main(
            [
                "generate",
                "--config",
                str(SMALL),
                "--seed",
                str(SEED),
                "--start",
                START,
                "--hours",
                "2",
                "--out",
                str(out),
            ]
        )
        == 0
    )
    assert sorted(p.name for p in out.iterdir()) == ["logs.jsonl", "manifest.json", "truth.jsonl", "world.json"]
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["seed"] == SEED and manifest["hours_generated"] == 2
    assert manifest["start"] == "2026-10-23T00:00:00.000000000Z"
    assert sum(manifest["records"].values()) == len((out / "logs.jsonl").read_text().splitlines())
    truth = [json.loads(line) for line in (out / "truth.jsonl").read_text().splitlines()]
    assert {t["expect"] for t in truth} == {"alert", "quiet"}


def test_generate_refuses_to_overwrite(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    args = [
        "generate",
        "--config",
        str(SMALL),
        "--seed",
        str(SEED),
        "--start",
        START,
        "--hours",
        "0",
        "--out",
        str(tmp_path),
    ]
    assert main(args) == 2
    assert "File exists" in capsys.readouterr().err


def test_start_must_be_utc() -> None:
    with pytest.raises(SystemExit):
        main(["generate", "--config", str(SMALL), "--seed", "1", "--start", "2026-10-23T00:00:00", "--out", "x"])


def test_stream_refuses_to_resume_a_different_run(tmp_path: Path) -> None:
    # A run that has already ended: if the identity check were missing, the
    # stream would fail fast on "the run ended" instead of waiting for a start.
    (tmp_path / "manifest.json").write_text(json.dumps({"seed": SEED + 1}))
    with pytest.raises(SystemExit, match="belongs to a different run"):
        main(
            [
                "stream",
                "--config",
                str(SMALL),
                "--seed",
                str(SEED),
                "--start",
                "2025-01-06T00:00:00Z",
                "--endpoint",
                "http://127.0.0.1:9",
                "--private-dir",
                str(tmp_path),
            ]
        )
