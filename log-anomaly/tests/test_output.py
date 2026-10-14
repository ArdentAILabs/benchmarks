from __future__ import annotations

import json
import threading
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import ClassVar

import pytest

from logsim.generate import Record
from logsim.output import OtlpHttpSink, SinkError, iso, otlp_payload

RECORDS = [
    Record(1_792_368_000_000_000_001, "INFO", "api", "POST /v1/branches 201", {"org.id": "org_a", "duration_ms": 80}),
    Record(1_792_368_000_500_000_000, "ERROR", "source-monitor", "source unreachable", {"ratio": 0.5}),
]


def test_iso_keeps_nanoseconds() -> None:
    assert iso(1_792_368_000_000_000_001) == "2026-10-19T00:00:00.000000001Z"


def test_otlp_payload_follows_the_json_encoding() -> None:
    payload = json.loads(otlp_payload(RECORDS, observed=7))
    resources = {r["resource"]["attributes"][0]["value"]["stringValue"]: r for r in payload["resourceLogs"]}
    assert set(resources) == {"api", "source-monitor"}
    (api,) = resources["api"]["scopeLogs"][0]["logRecords"]
    # OTLP/JSON writes 64-bit integers as decimal strings and enums as numbers.
    assert api["timeUnixNano"] == "1792368000000000001"
    assert api["observedTimeUnixNano"] == "7"
    assert api["severityNumber"] == 9
    assert api["attributes"] == [
        {"key": "org.id", "value": {"stringValue": "org_a"}},
        {"key": "duration_ms", "value": {"intValue": "80"}},
    ]
    (monitor,) = resources["source-monitor"]["scopeLogs"][0]["logRecords"]
    assert monitor["severityNumber"] == 17
    assert monitor["attributes"] == [{"key": "ratio", "value": {"doubleValue": 0.5}}]


class _Receiver(BaseHTTPRequestHandler):
    status = 200
    body = b""
    seen: ClassVar[list[tuple[str, str, dict[str, object]]]] = []

    def do_POST(self) -> None:
        length = int(self.headers["Content-Length"])
        type(self).seen.append((self.path, self.headers["Content-Type"], json.loads(self.rfile.read(length))))
        self.send_response(type(self).status)
        self.end_headers()
        self.wfile.write(type(self).body)

    def log_message(self, *args: object) -> None:
        pass


@pytest.fixture
def receiver() -> Iterator[tuple[str, type[_Receiver]]]:
    handler = type("Receiver", (_Receiver,), {"seen": []})
    server = HTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_port}", handler
    server.shutdown()


def test_sink_posts_to_v1_logs(receiver: tuple[str, type[_Receiver]]) -> None:
    url, handler = receiver
    OtlpHttpSink(url + "/", {}).send(RECORDS, observed=1)
    ((path, content_type, payload),) = handler.seen
    assert (path, content_type) == ("/v1/logs", "application/json")
    assert len(payload["resourceLogs"]) == 2


def test_sink_fails_on_http_error(receiver: tuple[str, type[_Receiver]]) -> None:
    url, handler = receiver
    handler.status, handler.body = 503, b"overloaded"
    with pytest.raises(SinkError, match="503"):
        OtlpHttpSink(url, {}).send(RECORDS, observed=1)


def test_sink_fails_on_partial_success(receiver: tuple[str, type[_Receiver]]) -> None:
    url, handler = receiver
    handler.body = json.dumps({"partialSuccess": {"rejectedLogRecords": "1", "errorMessage": "bad"}}).encode()
    with pytest.raises(SinkError, match="rejected 1 of 2"):
        OtlpHttpSink(url, {}).send(RECORDS, observed=1)


def test_sink_accepts_an_empty_partial_success(receiver: tuple[str, type[_Receiver]]) -> None:
    url, handler = receiver
    handler.body = b'{"partialSuccess":{}}'
    OtlpHttpSink(url, {}).send(RECORDS, observed=1)
