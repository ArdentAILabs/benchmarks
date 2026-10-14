"""The daily and weekly shape of a key's request rate, in the org's local time.

The plan needs it to seed retry cascades from the requests a key would send,
and the generator needs it to send them.
"""

from __future__ import annotations

from datetime import UTC, datetime
from zoneinfo import ZoneInfo

import numpy as np

from logsim.config import Traffic


def profile(seconds: np.ndarray, tz: ZoneInfo, traffic: Traffic) -> np.ndarray:
    """Relative request rate at each UTC second: a daily cosine and a weekend dip."""
    local = seconds + _offsets(tz, seconds)
    hour = (local % 86400) / 3600.0
    daily = 1.0 + traffic.diurnal_amplitude * np.cos(2.0 * np.pi * (hour - traffic.diurnal_peak_hour) / 24.0)
    weekday = (local // 86400 + 3) % 7  # 1970-01-01 was a Thursday; Monday = 0
    return daily * np.where(weekday >= 5, traffic.weekend_factor, 1.0)


def _offsets(tz: ZoneInfo, seconds: np.ndarray) -> np.ndarray:
    """UTC offset at each second, looked up once per UTC hour. Every offset
    change in a supported zone lands on a UTC hour; one that does not is refused
    rather than smeared across the hour."""
    hours = seconds // 3600
    offsets = np.empty_like(seconds)
    for hour in np.unique(hours).tolist():
        first, last = hour * 3600, hour * 3600 + 3599
        offset = _utc_offset(tz, first)
        if _utc_offset(tz, last) != offset:
            raise ValueError(f"{tz.key} changes UTC offset inside the hour at {first}; chunks must not straddle one")
        offsets[hours == hour] = offset
    return offsets


def _utc_offset(tz: ZoneInfo, second: int) -> int:
    delta = datetime.fromtimestamp(second, UTC).astimezone(tz).utcoffset()
    if delta is None:
        raise ValueError(f"{tz.key} has no UTC offset")
    return int(delta.total_seconds())
