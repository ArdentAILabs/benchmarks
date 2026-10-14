"""Command line: `logsim generate` writes a run to files, `logsim stream` emits it live."""

from __future__ import annotations

import argparse
import hashlib
import json
import signal
import sys
from collections import Counter
from datetime import datetime, timedelta
from pathlib import Path

from logsim import config, live
from logsim.generate import generate_chunk
from logsim.output import OtlpHttpSink, SinkError, iso, record_line, truth_line, world_document
from logsim.plan import NS, Plan, PlanError, build_plan
from logsim.repro import versions
from logsim.world import World, build_world


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="logsim", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    def common(sub: argparse.ArgumentParser) -> None:
        sub.add_argument("--config", type=Path, required=True)
        sub.add_argument("--seed", type=int, required=True)
        sub.add_argument(
            "--start", type=_utc_hour, required=True, help="run start, UTC, on the hour: 2026-10-19T00:00:00Z"
        )

    generate = commands.add_parser("generate", help="write logs, truth and world to a new directory")
    common(generate)
    generate.add_argument("--out", type=Path, required=True, help="must not exist")
    generate.add_argument(
        "--hours", type=int, help="generate only the first N hours of logs; truth always covers the run"
    )
    generate.set_defaults(handler=_generate)

    stream = commands.add_parser("stream", help="emit logs in real time to an OTLP/HTTP receiver")
    common(stream)
    stream.add_argument("--endpoint", required=True, help="OTLP/HTTP base URL; /v1/logs is appended")
    stream.add_argument(
        "--private-dir", type=Path, required=True, help="truth, world, manifest and journal; never shared"
    )
    stream.add_argument("--header", action="append", default=[], metavar="NAME=VALUE")
    stream.set_defaults(handler=_stream)

    args = parser.parse_args(argv)
    try:
        return args.handler(args)
    except (config.ConfigError, PlanError, SinkError, live.FellBehind, FileExistsError) as exc:
        print(f"logsim: {exc}", file=sys.stderr)
        return 2


def _utc_hour(text: str) -> int:
    try:
        moment = datetime.fromisoformat(text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc
    if moment.utcoffset() != timedelta(0):
        raise argparse.ArgumentTypeError("--start must carry a UTC offset, e.g. 2026-10-19T00:00:00Z")
    return int(moment.timestamp()) * NS


def _load(args: argparse.Namespace) -> tuple[config.Config, World, Plan, dict[str, object]]:
    cfg = config.load(args.config)
    world = build_world(cfg, args.seed)
    plan = build_plan(cfg, world, args.seed, args.start)
    identity = {
        "seed": args.seed,
        "config": args.config.name,
        "config_sha256": cfg.sha256,
        "start": iso(plan.start),
        "scored_from": iso(plan.scored_from),
        "end": iso(plan.end),
        "versions": versions(),
    }
    return cfg, world, plan, identity


def _write_static(directory: Path, world: World, plan: Plan) -> None:
    (directory / "world.json").write_text(json.dumps(world_document(world), indent=2) + "\n", encoding="utf-8")
    with (directory / "truth.jsonl").open("w", encoding="utf-8") as handle:
        for truth in plan.truth:
            handle.write(truth_line(truth, world) + "\n")


def _generate(args: argparse.Namespace) -> int:
    cfg, world, plan, identity = _load(args)
    chunks = plan.chunks if args.hours is None else min(max(args.hours, 0), plan.chunks)
    args.out.mkdir(parents=True, exist_ok=False)
    _write_static(args.out, world, plan)
    digest = hashlib.sha256()
    counts: Counter[str] = Counter()
    with (args.out / "logs.jsonl").open("w", encoding="utf-8") as handle:
        for index in range(chunks):
            for record in generate_chunk(cfg, world, plan, args.seed, index):
                line = record_line(record) + "\n"
                handle.write(line)
                digest.update(line.encode())
                counts[f"{record.service} {record.severity}"] += 1
    manifest = {
        **identity,
        "hours_generated": chunks,
        "records": dict(sorted(counts.items())),
        "logs_sha256": digest.hexdigest(),
    }
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    _summarise(plan, sum(counts.values()), chunks)
    return 0


def _stream(args: argparse.Namespace) -> int:
    cfg, world, plan, identity = _load(args)
    headers = {}
    for header in args.header:
        name, sep, value = header.partition("=")
        if not sep or not name:
            raise SystemExit(f"logsim: --header {header!r} is not NAME=VALUE")
        headers[name] = value
    args.private_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.private_dir / "manifest.json"
    if manifest_path.exists():
        # A restart must be the same run, or the resumed stream would continue
        # a different plan under the old truth file.
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if existing != identity:
            raise SystemExit(f"logsim: {args.private_dir} belongs to a different run; refusing to resume")
    else:
        _write_static(args.private_dir, world, plan)
        manifest_path.write_text(json.dumps(identity, indent=2) + "\n", encoding="utf-8")
    _summarise(plan, None, plan.chunks)
    # docker stop and systemd send SIGTERM; turning it into SystemExit lets the
    # stream journal where it stopped, so the gap before a restart is exact.
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(143))
    live.stream(cfg, world, plan, args.seed, OtlpHttpSink(args.endpoint, headers), args.private_dir / "journal.jsonl")
    return 0


def _summarise(plan: Plan, records: int | None, hours: int) -> None:
    scored = Counter(f"{t.expect:5} {t.kind}" for t in plan.truth if t.scored)
    print(f"run {iso(plan.start)} .. {iso(plan.end)}, scored from {iso(plan.scored_from)}")
    if records is not None:
        print(f"{records} records over {hours} hours")
    print("scored truth:")
    for label, count in sorted(scored.items()):
        print(f"  {count:5d}  {label}")


if __name__ == "__main__":
    sys.exit(main())
