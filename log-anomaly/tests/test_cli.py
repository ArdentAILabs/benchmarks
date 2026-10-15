from __future__ import annotations

import json
from pathlib import Path

import pytest
from conftest import SEED, SMALL

from logsim.cli import MIN_STREAM_SEED, main

START = "2026-10-23T00:00:00Z"
STREAM_SEED = 2**127 + 2  # what `logsim seed` prints is this size


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


def _stream_args(seed: int, private_dir: Path) -> list[str]:
    # A run that has already ended, so a stream that got past its checks
    # fails fast on "the run ended" instead of waiting for a start.
    return [
        "stream",
        "--config",
        str(SMALL),
        "--seed",
        str(seed),
        "--start",
        "2025-01-06T00:00:00Z",
        "--endpoint",
        "http://127.0.0.1:9",
        "--private-dir",
        str(private_dir),
    ]


def test_stream_refuses_to_resume_a_different_run(tmp_path: Path) -> None:
    (tmp_path / "manifest.json").write_text(json.dumps({"seed": STREAM_SEED + 1}))
    with pytest.raises(SystemExit, match="belongs to a different run"):
        main(_stream_args(STREAM_SEED, tmp_path))


def test_stream_refuses_a_guessable_seed(tmp_path: Path) -> None:
    private = tmp_path / "private"
    with pytest.raises(SystemExit, match="cannot be guessed"):
        main(_stream_args(MIN_STREAM_SEED - 1, private))
    assert not private.exists(), "refused before writing truth"


def test_seed_prints_a_fresh_128_bit_seed(capsys: pytest.CaptureFixture[str]) -> None:
    seeds = []
    for _ in range(2):
        assert main(["seed"]) == 0
        seeds.append(int(capsys.readouterr().out))
    assert seeds[0] != seeds[1]
    assert all(2**127 <= s < 2**128 for s in seeds)
