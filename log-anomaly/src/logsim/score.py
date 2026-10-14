"""Score one system's pages against a run's truth.

Input is the pages that reached a person, one JSON object per line:
`time` (RFC 3339 with an offset) and, where the page names one, `api_key.id`
and/or `org.id`. A page may also say what it saw, as `signal` (one of
`logsim.kinds.SIGNAL_NAMES`); then it matches only cases showing that signal.
Either every page in a file has a signal or none does, so two systems are never
scored by different rules without it showing. Every page is classified exactly
once:

- true: names the entity of a must-alert case and arrives between the case's
  start and `deadline` after its end. The first such page detects the case.
- duplicate: would be true, but the case was already detected. Reported per
  detected case, since one incident paging thirty times is its own failure.
- false: everything else in the scored phase. A false page that lands on a
  must-stay-quiet case is charged to that case's kind; one that names an
  entity but matches nothing is `unexplained`; one that names nothing is
  `unattributed`.
- excluded: outside the scored phase, inside a stream gap, or matching only
  cases that are themselves excluded.

A case counts only if it starts in the scored phase and the whole span from its
start to `deadline` after its end was emitted: stream gaps and unemitted hours
are time no detector saw.
"""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from logsim.kinds import SCOPE, SIGNAL_NAMES, SIGNALS
from logsim.plan import HOUR, MINUTE, NS

WEEK = 7 * 24 * HOUR

_TIME = re.compile(r"^(\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d)(?:\.(\d{1,9}))?(Z|[+-]\d\d:\d\d)$")


class ScoreError(ValueError):
    pass


@dataclass(frozen=True)
class Case:
    id: str
    kind: str
    expect: str
    scored: bool
    org: str | None
    key: str | None
    start: int
    end: int
    member_keys: frozenset[str]
    member_orgs: frozenset[str]


@dataclass(frozen=True)
class Page:
    line: int
    time: int
    org: str | None  # resolved from the key when the page names only a key
    key: str | None
    signal: str | None = None


@dataclass(frozen=True)
class Run:
    scored_from: int
    horizon: int  # nothing at or after this was emitted
    gaps: tuple[tuple[int, int], ...]
    cases: tuple[Case, ...]
    key_org: dict[str, str]
    orgs: frozenset[str]

    def observed(self, start: int, end: int) -> bool:
        """Was every instant of [start, end] emitted?"""
        return end < self.horizon and not any(s <= end and start < e for s, e in self.gaps)

    def observed_scored_time(self) -> int:
        total = max(self.horizon - self.scored_from, 0)
        for s, e in self.gaps:
            total -= max(0, min(e, self.horizon) - max(s, self.scored_from))
        return total


@dataclass
class Report:
    deadline_minutes: float
    observed_weeks: float
    signals: bool  # pages were matched on signal as well as entity and time
    pages: Counter[str] = field(default_factory=Counter)
    false_by_cause: Counter[str] = field(default_factory=Counter)
    detected: dict[str, int] = field(default_factory=dict)  # case id -> ns from start to first page
    case_pages: Counter[str] = field(default_factory=Counter)  # detected case id -> true and duplicate pages
    must_alert: list[Case] = field(default_factory=list)
    must_stay_quiet: list[Case] = field(default_factory=list)
    paged_quiet: set[str] = field(default_factory=set)
    false_pages: list[dict[str, object]] = field(default_factory=list)

    def _detected_by_kind(self) -> dict[str, list[str]]:
        by_kind: dict[str, list[str]] = {}
        for case in self.must_alert:
            if case.id in self.detected:
                by_kind.setdefault(case.kind, []).append(case.id)
        return by_kind

    def as_dict(self) -> dict[str, object]:
        detected = len(self.detected)
        cases = len(self.must_alert)
        true, false = self.pages["true"], self.pages["false"]
        ttd = sorted(self.detected.values())
        return {
            "deadline_minutes": self.deadline_minutes,
            "matched_on": "entity, time and signal" if self.signals else "entity and time",
            "observed_weeks": round(self.observed_weeks, 3),
            "pages": dict(sorted(self.pages.items())),
            "false_pages_per_week": round(false / self.observed_weeks, 2) if self.observed_weeks else None,
            "false_by_cause": dict(self.false_by_cause.most_common()),
            "precision": round(true / (true + false), 4) if true + false else None,
            "recall": {
                "detected": detected,
                "cases": cases,
                "rate": round(detected / cases, 4) if cases else None,
                "wilson_95": [round(x, 4) for x in wilson(detected, cases)] if cases else None,
            },
            "recall_by_kind": {
                kind: {
                    "detected": sum(c.id in self.detected for c in self.must_alert if c.kind == kind),
                    "cases": sum(c.kind == kind for c in self.must_alert),
                }
                for kind in sorted({c.kind for c in self.must_alert})
            },
            "quiet_cases_paged": {
                kind: {
                    "paged": sum(c.id in self.paged_quiet for c in self.must_stay_quiet if c.kind == kind),
                    "cases": sum(c.kind == kind for c in self.must_stay_quiet),
                }
                for kind in sorted({c.kind for c in self.must_stay_quiet})
            },
            "pages_per_detected_case": {
                kind: round(sum(self.case_pages[i] for i in ids) / len(ids), 2)
                for kind, ids in sorted(self._detected_by_kind().items())
            },
            "minutes_to_detect": (
                {"median": round(ttd[len(ttd) // 2] / MINUTE, 2), "max": round(ttd[-1] / MINUTE, 2)} if ttd else None
            ),
            "missed": [{"id": c.id, "kind": c.kind} for c in self.must_alert if c.id not in self.detected],
            "false_page_list": self.false_pages,
        }


def parse_time(text: object) -> int:
    if not isinstance(text, str) or not (match := _TIME.match(text)):
        raise ScoreError(f"time {text!r} is not RFC 3339 with an offset")
    whole, fraction, offset = match.groups()
    moment = datetime.fromisoformat(whole + ("+00:00" if offset == "Z" else offset))
    return int(moment.timestamp()) * NS + int((fraction or "").ljust(9, "0"))


def wilson(successes: int, trials: int, z: float = 1.959964) -> tuple[float, float]:
    p = successes / trials
    centre = p + z * z / (2 * trials)
    spread = z * math.sqrt(p * (1 - p) / trials + z * z / (4 * trials * trials))
    denominator = 1 + z * z / trials
    return max(0.0, (centre - spread) / denominator), min(1.0, (centre + spread) / denominator)


def load_run(directory: Path, allow_unfinished: bool = False) -> Run:
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    start = parse_time(manifest["start"])
    end = parse_time(manifest["end"])
    journal = directory / "journal.jsonl"
    if journal.exists():
        gaps, horizon = _journal(journal, start, end, allow_unfinished)
    elif "hours_generated" in manifest:
        gaps, horizon = (), min(end, start + int(manifest["hours_generated"]) * HOUR)
    else:
        raise ScoreError(f"{directory}: neither a stream journal nor a generate manifest")

    world = json.loads((directory / "world.json").read_text(encoding="utf-8"))
    key_org = {key["id"]: org["id"] for org in world["orgs"] for key in org["keys"]}
    cases = []
    for number, line in enumerate((directory / "truth.jsonl").read_text(encoding="utf-8").splitlines(), start=1):
        doc = json.loads(line)
        if doc["kind"] not in SCOPE:
            raise ScoreError(f"truth.jsonl:{number}: no matching rule for kind {doc['kind']!r}")
        detail = doc["detail"]
        if SCOPE[doc["kind"]] == "members" and "orgs" not in detail:
            raise ScoreError(f"truth.jsonl:{number}: {doc['kind']} case lists no member orgs")
        cases.append(
            Case(
                id=doc["id"],
                kind=doc["kind"],
                expect=doc["expect"],
                scored=doc["scored"],
                org=doc["org.id"],
                key=doc["api_key.id"],
                start=parse_time(doc["start"]),
                end=parse_time(doc["end"]),
                member_keys=frozenset(detail.get("keys", ())),
                member_orgs=frozenset(detail.get("orgs", ())),
            )
        )
    return Run(
        scored_from=parse_time(manifest["scored_from"]),
        horizon=horizon,
        gaps=tuple(gaps),
        cases=tuple(cases),
        key_org=key_org,
        orgs=frozenset(org["id"] for org in world["orgs"]),
    )


def _journal(path: Path, start: int, end: int, allow_unfinished: bool) -> tuple[list[tuple[int, int]], int]:
    """Gaps between what was sent and where the next start resumed."""
    covered = start
    gaps: list[tuple[int, int]] = []
    finished = False
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        event = json.loads(line)
        kind = event["event"]
        if kind == "emitting":
            resumed = parse_time(event["emitting_from"])
            if resumed > covered:
                gaps.append((covered, resumed))
            covered = resumed
            finished = False
        elif kind in ("progress", "stopped"):
            covered = parse_time(event["sent_through"])
        elif kind == "finished":
            finished = True
            covered = end
        else:
            raise ScoreError(f"{path.name}:{number}: unknown event {kind!r}")
    if not finished and not allow_unfinished:
        raise ScoreError(
            f"{path.name} does not show the run finishing; pass --allow-unfinished to score what was sent, "
            f"up to the last record journalled"
        )
    return gaps, covered


def load_pages(path: Path, run: Run) -> list[Page]:
    pages = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        where = f"{path.name}:{number}"
        try:
            doc = json.loads(line)
        except ValueError as exc:
            raise ScoreError(f"{where}: {exc}") from exc
        if not isinstance(doc, dict):
            raise ScoreError(f"{where}: expected a JSON object")
        try:
            time = parse_time(doc.get("time"))
        except ScoreError as exc:
            raise ScoreError(f"{where}: {exc}") from exc
        key, org = doc.get("api_key.id"), doc.get("org.id")
        # An id the world does not know is a typo or a mapping bug in whatever
        # produced the file; matching it silently would turn it into a false page.
        signal = doc.get("signal")
        if signal is not None and signal not in SIGNAL_NAMES:
            raise ScoreError(f"{where}: unknown signal {signal!r}; expected one of {sorted(SIGNAL_NAMES)}")
        if key is not None and key not in run.key_org:
            raise ScoreError(f"{where}: unknown api_key.id {key!r}")
        if org is not None and org not in run.orgs:
            raise ScoreError(f"{where}: unknown org.id {org!r}")
        if key is not None and org is not None and run.key_org[key] != org:
            raise ScoreError(f"{where}: api_key.id {key!r} belongs to {run.key_org[key]!r}, not {org!r}")
        pages.append(Page(number, time, org if org is not None else run.key_org.get(key) if key else None, key, signal))
    return pages


def _names(page: Page, case: Case) -> bool:
    if page.signal is not None and page.signal not in SIGNALS[case.kind]:
        return False
    scope = SCOPE[case.kind]
    if scope == "key":
        return page.key is not None and page.key == case.key
    if scope == "org":
        return page.org is not None and page.org == case.org
    if page.key is None and page.org is None:
        return True
    return page.key in case.member_keys or page.org in case.member_orgs


def score(run: Run, pages: list[Page], deadline_minutes: float) -> Report:
    deadline = int(deadline_minutes * MINUTE)
    countable = {c.id for c in run.cases if c.scored and run.observed(c.start, c.end + deadline)}
    signals = {p.signal is not None for p in pages}
    if len(signals) > 1:
        raise ScoreError("some pages carry a signal and some do not; label all of them or none")
    report = Report(deadline_minutes, run.observed_scored_time() / WEEK, signals == {True})
    report.must_alert = [c for c in run.cases if c.id in countable and c.expect == "alert"]
    report.must_stay_quiet = [c for c in run.cases if c.id in countable and c.expect == "quiet"]

    for page in sorted(pages, key=lambda p: (p.time, p.line)):
        if page.time < run.scored_from or not run.observed(page.time, page.time):
            report.pages["excluded"] += 1
            continue
        hits = [c for c in run.cases if c.start <= page.time <= c.end + deadline and _names(page, c)]
        must = [c for c in hits if c.expect == "alert"]
        quiet = [c for c in hits if c.expect == "quiet" and c.id in countable]
        if must:
            live = sorted((c for c in must if c.id in countable), key=lambda c: c.start)
            if not live:
                report.pages["excluded"] += 1
            elif fresh := [c for c in live if c.id not in report.detected]:
                report.detected[fresh[0].id] = page.time - fresh[0].start
                report.case_pages[fresh[0].id] += 1
                report.pages["true"] += 1
            else:
                report.case_pages[live[0].id] += 1
                report.pages["duplicate"] += 1
            continue
        if hits and not quiet:
            report.pages["excluded"] += 1
            continue
        if quiet:
            first = min(quiet, key=lambda c: c.start)
            cause, case = first.kind, first.id
            report.paged_quiet.update(c.id for c in quiet)
        else:
            cause = "unattributed" if page.key is None and page.org is None else "unexplained"
            case = None
        report.pages["false"] += 1
        report.false_by_cause[cause] += 1
        report.false_pages.append({"line": page.line, "cause": cause, "case": case})
    return report


def render(summary: dict[str, object]) -> str:
    pages = summary["pages"]
    recall = summary["recall"]
    lines = [
        f"observed scored time: {summary['observed_weeks']} weeks; deadline {summary['deadline_minutes']} min",
        f"pages matched on {summary['matched_on']}",
        f"pages: {sum(pages.values())} ({', '.join(f'{k} {v}' for k, v in pages.items())})",
        f"false pages per week: {summary['false_pages_per_week']}",
    ]
    lines += [f"  {count:5d}  {cause}" for cause, count in summary["false_by_cause"].items()]
    lines.append(f"precision: {summary['precision']}")
    if recall["cases"]:
        low, high = recall["wilson_95"]
        lines.append(
            f"recall: {recall['detected']}/{recall['cases']} = {recall['rate']} (95% Wilson interval {low}-{high})"
        )
    lines += [f"  {v['detected']:3d}/{v['cases']:<3d} {kind}" for kind, v in summary["recall_by_kind"].items()]
    lines.append("quiet cases that drew a page:")
    lines += [f"  {v['paged']:4d}/{v['cases']:<4d} {kind}" for kind, v in summary["quiet_cases_paged"].items()]
    if summary["pages_per_detected_case"]:
        lines.append("pages per detected case:")
        lines += [f"  {v:7.2f}  {kind}" for kind, v in summary["pages_per_detected_case"].items()]
    if summary["minutes_to_detect"]:
        ttd = summary["minutes_to_detect"]
        lines.append(f"minutes to detect: median {ttd['median']}, max {ttd['max']}")
    if summary["missed"]:
        lines.append("missed: " + ", ".join(f"{m['id']} {m['kind']}" for m in summary["missed"]))
    return "\n".join(lines)
