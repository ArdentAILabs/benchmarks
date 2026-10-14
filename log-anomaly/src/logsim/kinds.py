"""Every case kind: which entity a page must name to match it, and which
signals it shows up in. The plan keeps an alert case clear of quiet cases it
shares a signal with, and the scorer matches pages by the same tables."""

from __future__ import annotations

# "members" cases span several orgs and match a page naming any key or org
# involved, or none.
SCOPE = {
    "burst_on_steady_key": "key",
    "natural_burst": "key",
    "key_silence": "key",
    "novel_error": "key",
    "unscheduled_bulk_export": "key",
    "scheduled_export": "key",
    "retry_storm": "key",
    "retry_echo": "key",
    "error_rate_shift": "key",
    "batch_surge": "key",
    "unplanned_outage": "org",
    "maintenance_overrun": "org",
    "maintenance_window": "org",
    "onboarding": "org",
    "latency_leak": "org",
    "organic_growth": "org",
    "chance_error_clump": "members",
    "renamed_error": "members",
    "provider_outage": "members",
    "bad_deploy": "members",
}

REQUESTS, ERRORS, LATENCY, HEALTH, EXPORTS = "requests", "errors", "latency", "health", "exports"
SIGNAL_NAMES = frozenset({REQUESTS, ERRORS, LATENCY, HEALTH, EXPORTS})

SIGNALS: dict[str, frozenset[str]] = {
    "burst_on_steady_key": frozenset({REQUESTS}),
    "natural_burst": frozenset({REQUESTS}),
    "key_silence": frozenset({REQUESTS}),
    "novel_error": frozenset({ERRORS}),
    "unscheduled_bulk_export": frozenset({EXPORTS}),
    "scheduled_export": frozenset({EXPORTS}),
    "retry_storm": frozenset({REQUESTS, ERRORS}),
    "retry_echo": frozenset({REQUESTS, ERRORS}),
    "error_rate_shift": frozenset({ERRORS}),
    "batch_surge": frozenset({REQUESTS, ERRORS}),
    "unplanned_outage": frozenset({HEALTH}),
    "maintenance_overrun": frozenset({HEALTH}),
    "maintenance_window": frozenset({HEALTH}),
    "onboarding": frozenset({REQUESTS, HEALTH}),
    "latency_leak": frozenset({LATENCY}),
    "organic_growth": frozenset({REQUESTS}),
    "chance_error_clump": frozenset({ERRORS}),
    "renamed_error": frozenset({ERRORS}),
    "provider_outage": frozenset({HEALTH}),
    "bad_deploy": frozenset({ERRORS}),
}

if SCOPE.keys() != SIGNALS.keys():
    raise AssertionError(f"kinds without both a scope and signals: {sorted(SCOPE.keys() ^ SIGNALS.keys())}")
