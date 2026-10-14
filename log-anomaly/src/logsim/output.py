"""Serialise records, ground truth and the world, and deliver records over OTLP/HTTP."""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from datetime import UTC, datetime

from logsim.generate import Attribute, Record
from logsim.plan import NS, Truth
from logsim.world import World

SEVERITY_NUMBER = {"INFO": 9, "WARN": 13, "ERROR": 17}  # OpenTelemetry log data model


def iso(ns: int) -> str:
    whole, fraction = divmod(ns, NS)
    return datetime.fromtimestamp(whole, UTC).strftime("%Y-%m-%dT%H:%M:%S") + f".{fraction:09d}Z"


def record_line(record: Record) -> str:
    return json.dumps(
        {
            "time": iso(record.time),
            "time_unix_nano": record.time,
            "severity": record.severity,
            "service": record.service,
            "body": record.body,
            "attributes": record.attributes,
        },
        separators=(",", ":"),
    )


def truth_line(truth: Truth, world: World) -> str:
    return json.dumps(
        {
            "id": truth.id,
            "kind": truth.kind,
            "pair": truth.pair,
            "expect": truth.expect,
            "scored": truth.scored,
            "org.id": None if truth.org is None else world.orgs[truth.org].id,
            "api_key.id": None if truth.key is None else world.keys[truth.key].id,
            "start": iso(truth.start),
            "end": iso(truth.end),
            "detail": truth.detail,
        },
        separators=(",", ":"),
    )


def world_document(world: World) -> dict[str, object]:
    return {
        "orgs": [
            {
                "id": org.id,
                "timezone": org.timezone,
                "provider": org.provider.name,
                "maintenance_minute": org.maintenance_minute,
                "fixed": org.fixed,
                "keys": [
                    {
                        "id": world.keys[k].id,
                        "archetype": world.keys[k].archetype,
                        "rate_per_second": world.keys[k].rate,
                        "gamma_shape": world.keys[k].gamma_shape,
                        "weekly_export": (
                            None
                            if world.keys[k].weekly_export is None
                            else {
                                "weekday": world.keys[k].weekly_export.weekday,
                                "minute_of_day": world.keys[k].weekly_export.minute_of_day,
                            }
                        ),
                    }
                    for k in org.keys
                ],
            }
            for org in world.orgs
        ]
    }


def otlp_payload(records: list[Record], observed: int) -> bytes:
    """An ExportLogsServiceRequest in OTLP/JSON: 64-bit integers as decimal
    strings, enums as numbers, one resource per service."""
    by_service: dict[str, list[dict[str, object]]] = {}
    for r in records:
        by_service.setdefault(r.service, []).append(
            {
                "timeUnixNano": str(r.time),
                "observedTimeUnixNano": str(observed),
                "severityNumber": SEVERITY_NUMBER[r.severity],
                "severityText": r.severity,
                "body": {"stringValue": r.body},
                "attributes": [_key_value(k, v) for k, v in r.attributes.items()],
            }
        )
    payload = {
        "resourceLogs": [
            {
                "resource": {"attributes": [_key_value("service.name", service)]},
                "scopeLogs": [{"scope": {"name": "logsim"}, "logRecords": logs}],
            }
            for service, logs in by_service.items()
        ]
    }
    return json.dumps(payload, separators=(",", ":")).encode()


def _key_value(key: str, value: Attribute) -> dict[str, object]:
    if isinstance(value, bool):
        raise TypeError(f"attribute {key}: booleans are not emitted")
    if isinstance(value, int):
        return {"key": key, "value": {"intValue": str(value)}}
    if isinstance(value, float):
        return {"key": key, "value": {"doubleValue": value}}
    return {"key": key, "value": {"stringValue": value}}


class SinkError(RuntimeError):
    pass


class OtlpHttpSink:
    """POSTs batches to an OTLP/HTTP receiver. Any rejection is fatal: a record
    one system received and the other did not would void the comparison."""

    def __init__(self, endpoint: str, headers: dict[str, str], timeout_seconds: float = 10.0):
        if not endpoint.startswith(("http://", "https://")):
            raise ValueError("endpoint must be an http(s) URL")
        self.url = endpoint.rstrip("/") + "/v1/logs"
        self.headers = {"Content-Type": "application/json", **headers}
        self.timeout = timeout_seconds

    def send(self, records: list[Record], observed: int) -> None:
        request = urllib.request.Request(
            self.url, data=otlp_payload(records, observed), headers=self.headers, method="POST"
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                body = response.read()
        except urllib.error.HTTPError as exc:
            raise SinkError(f"{self.url} returned {exc.code}: {exc.read()[:500]!r}") from exc
        if not body:
            return
        try:
            document = json.loads(body)
        except ValueError as exc:
            raise SinkError(f"{self.url} returned a non-JSON body: {body[:200]!r}") from exc
        partial = document.get("partialSuccess") or {}
        rejected = int(partial.get("rejectedLogRecords", 0) or 0)
        if rejected:
            raise SinkError(
                f"{self.url} rejected {rejected} of {len(records)} records: {partial.get('errorMessage')!r}"
            )
