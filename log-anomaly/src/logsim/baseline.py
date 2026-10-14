"""A threshold baseline: what simple per-series rules score on a run.

It is the control the other systems have to beat. If it scores well, the
benchmark is too easy. Thresholds are learned from the warm-up and then frozen;
nothing adapts during the scored phase. One rule per signal:

- request spike: a key's requests in a minute exceed 3x its warm-up maximum
  (at least 10)
- silence: a key that averaged at least one request a minute in warm-up sends
  none for 30 minutes
- source down: an org's source fails two health checks in a row
- error volume: 4 lines of one error message (digits and ids masked) within
  10 minutes, fleet-wide
- export volume: a key's exported bytes over the trailing hour exceed 3x its
  warm-up maximum (at least 1 GB)
- server errors: a key's 5xx responses over the trailing 10 minutes exceed 3x
  its warm-up maximum (at least 5)
- latency: an org's mean request latency over the trailing 10 minutes, with at
  least 20 requests, exceeds 2x its warm-up maximum

Each rule pages at most once per entity per 30 minutes; the source-down rule
re-arms only after a healthy check. Every page carries the signal its rule
watches, so it can be scored with or without signal matching.
"""

from __future__ import annotations

import json
import re
from collections import defaultdict, deque
from collections.abc import Iterator
from pathlib import Path

from logsim.kinds import ERRORS, EXPORTS, HEALTH, LATENCY, REQUESTS
from logsim.output import iso
from logsim.plan import MINUTE
from logsim.score import parse_time

_SPIKE_FACTOR = 3.0
_SPIKE_FLOOR = 10
_SILENCE = 30 * MINUTE
_SILENCE_MIN_PER_MINUTE = 1.0
_ERROR_COUNT = 4
_ERROR_WINDOW = 10 * MINUTE
_EXPORT_FACTOR = 3.0
_EXPORT_FLOOR = 1 << 30
_EXPORT_WINDOW = 60 * MINUTE
_COOLDOWN = 30 * MINUTE
_TRAILING_MINUTES = 10
_SERVER_ERROR_FACTOR = 3.0
_SERVER_ERROR_FLOOR = 5
_LATENCY_FACTOR = 2.0
_LATENCY_MIN_REQUESTS = 20
_MASK = re.compile(r"\b[0-9a-f]{8}\b|\d+")


def mask(body: str) -> str:
    return _MASK.sub("#", body)


class ThresholdBaseline:
    def __init__(self, start: int, scored_from: int):
        self.start = start
        self.scored_from = scored_from
        self.pages: list[dict[str, object]] = []
        self._last_page: dict[tuple[str, str], int] = {}

        self._minute: int | None = None
        self._counts: dict[str, int] = defaultdict(int)  # key -> requests this minute
        self._warm_max: dict[str, int] = defaultdict(int)
        self._warm_total: dict[str, int] = defaultdict(int)
        self._last_request: dict[str, int] = {}
        self._silent_paged: set[str] = set()
        self._key_org: dict[str, str] = {}

        self._failures: dict[str, int] = defaultdict(int)
        self._errors: dict[str, deque[tuple[int, str]]] = defaultdict(deque)
        self._exports: dict[str, deque[tuple[int, int]]] = defaultdict(deque)
        self._export_sum: dict[str, int] = defaultdict(int)
        self._export_warm_max: dict[str, int] = defaultdict(int)

        # Per-minute tallies over the trailing window, oldest first.
        self._server_errors: dict[str, deque[int]] = defaultdict(deque)
        self._server_now: dict[str, int] = defaultdict(int)
        self._server_warm_max: dict[str, int] = defaultdict(int)
        self._latency: dict[str, deque[tuple[int, int]]] = defaultdict(deque)  # (requests, total ms)
        self._latency_now: dict[str, list[int]] = defaultdict(lambda: [0, 0])
        self._latency_warm_max: dict[str, float] = defaultdict(float)

    def feed(self, record: dict[str, object]) -> None:
        t = int(record["time_unix_nano"])
        attributes = record["attributes"]
        minute = t // MINUTE
        if self._minute is not None and minute != self._minute:
            self._close_minutes(self._minute, minute)
        self._minute = minute
        key = attributes.get("api_key.id")
        if key is not None:
            self._key_org[key] = attributes["org.id"]

        if "http.route" in attributes:
            self._counts[key] += 1
            self._last_request[key] = t
            self._silent_paged.discard(key)
            if attributes["http.response.status_code"] >= 500:
                self._server_now[key] += 1
            tally = self._latency_now[attributes["org.id"]]
            tally[0] += 1
            tally[1] += attributes["duration_ms"]
        elif record["service"] == "source-monitor":
            self._health(t, attributes["org.id"], record["severity"] == "ERROR")
        elif "export.bytes" in attributes:
            self._export(t, key, int(attributes["export.bytes"]))
        elif record["severity"] in ("WARN", "ERROR"):
            self._error(t, key, mask(record["body"]))

    def finish(self) -> list[dict[str, object]]:
        if self._minute is not None:
            self._close_minutes(self._minute, self._minute + 1)
        return self.pages

    def _close_minutes(self, finished: int, current: int) -> None:
        end = (finished + 1) * MINUTE
        warm = end <= self.scored_from
        for key, count in self._counts.items():
            if warm:
                self._warm_max[key] = max(self._warm_max[key], count)
                self._warm_total[key] += count
            elif count > max(_SPIKE_FACTOR * self._warm_max.get(key, 0), _SPIKE_FLOOR):
                self._page(end, "request_spike", REQUESTS, key=key, summary=f"{count} requests in a minute")
        self._counts.clear()
        if current != finished + 1:
            # Health checks every 30 s guarantee a record every minute; the
            # trailing windows count minutes by closing them, so a gap would
            # silently stretch them.
            raise ValueError(f"no records between minute {finished} and {current}; the baseline needs one a minute")
        self._close_server_errors(end, warm)
        self._close_latency(end, warm)
        if warm:
            return
        warm_minutes = (self.scored_from - self.start) / MINUTE
        for key, last in self._last_request.items():
            busy = self._warm_total.get(key, 0) / warm_minutes >= _SILENCE_MIN_PER_MINUTE
            if busy and key not in self._silent_paged and current * MINUTE - last >= _SILENCE:
                self._silent_paged.add(key)
                self._page(current * MINUTE, "silence", REQUESTS, key=key, summary="no requests for 30 minutes")

    def _close_server_errors(self, end: int, warm: bool) -> None:
        for key in set(self._server_errors) | set(self._server_now):
            window = self._server_errors[key]
            window.append(self._server_now.pop(key, 0))
            if len(window) > _TRAILING_MINUTES:
                window.popleft()
            total = sum(window)
            if warm:
                self._server_warm_max[key] = max(self._server_warm_max[key], total)
            elif total > max(_SERVER_ERROR_FACTOR * self._server_warm_max.get(key, 0), _SERVER_ERROR_FLOOR):
                self._page(end, "server_errors", ERRORS, key=key, summary=f"{total} 5xx in 10 minutes")

    def _close_latency(self, end: int, warm: bool) -> None:
        for org in set(self._latency) | set(self._latency_now):
            window = self._latency[org]
            window.append(tuple(self._latency_now.pop(org, (0, 0))))
            if len(window) > _TRAILING_MINUTES:
                window.popleft()
            requests = sum(n for n, _ in window)
            if requests < _LATENCY_MIN_REQUESTS:
                continue
            mean = sum(ms for _, ms in window) / requests
            if warm:
                self._latency_warm_max[org] = max(self._latency_warm_max[org], mean)
            elif org in self._latency_warm_max and mean > _LATENCY_FACTOR * self._latency_warm_max[org]:
                self._page(end, "latency", LATENCY, org=org, summary=f"mean latency {mean:.0f} ms over 10 minutes")

    def _health(self, t: int, org: str, failed: bool) -> None:
        if not failed:
            self._failures[org] = 0
            return
        self._failures[org] += 1
        if self._failures[org] == 2 and t >= self.scored_from:
            self._page(t, "source_down", HEALTH, org=org, summary="two failed health checks", cooldown=False)

    def _error(self, t: int, key: str | None, message: str) -> None:
        window = self._errors[message]
        window.append((t, key or ""))
        while window and window[0][0] <= t - _ERROR_WINDOW:
            window.popleft()
        if len(window) >= _ERROR_COUNT and t >= self.scored_from:
            keys = {k for _, k in window}
            only = next(iter(keys)) if len(keys) == 1 and "" not in keys else None
            self._page(t, "error_volume", ERRORS, key=only, entity=message, summary=f"{len(window)}x {message}")

    def _export(self, t: int, key: str, size: int) -> None:
        window = self._exports[key]
        window.append((t, size))
        self._export_sum[key] += size
        while window and window[0][0] <= t - _EXPORT_WINDOW:
            self._export_sum[key] -= window.popleft()[1]
        total = self._export_sum[key]
        if t < self.scored_from:
            self._export_warm_max[key] = max(self._export_warm_max[key], total)
        elif total > max(_EXPORT_FACTOR * self._export_warm_max.get(key, 0), _EXPORT_FLOOR):
            self._page(t, "export_volume", EXPORTS, key=key, summary=f"{total} bytes exported in an hour")

    def _page(
        self,
        t: int,
        rule: str,
        signal: str,
        *,
        key: str | None = None,
        org: str | None = None,
        entity: str | None = None,
        summary: str,
        cooldown: bool = True,
    ) -> None:
        slot = (rule, entity or key or org or "")
        if cooldown and t - self._last_page.get(slot, -_COOLDOWN) < _COOLDOWN:
            return
        self._last_page[slot] = t
        page: dict[str, object] = {"time": iso(t), "rule": rule, "signal": signal, "summary": summary}
        if key is not None:
            page["api_key.id"] = key
            page["org.id"] = self._key_org[key]
        elif org is not None:
            page["org.id"] = org
        self.pages.append(page)


def run_baseline(directory: Path) -> list[dict[str, object]]:
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    baseline = ThresholdBaseline(parse_time(manifest["start"]), parse_time(manifest["scored_from"]))
    for record in _records(directory / "logs.jsonl"):
        baseline.feed(record)
    return baseline.finish()


def _records(path: Path) -> Iterator[dict[str, object]]:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            yield json.loads(line)
