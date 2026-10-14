from __future__ import annotations

from pathlib import Path

import pytest
from conftest import SMALL, variant

from logsim.config import ConfigError, load

DEV = Path(__file__).parents[1] / "configs" / "dev.toml"


def test_shipped_configs_load() -> None:
    assert load(DEV).scored_days == 14
    assert load(SMALL).scored_days == 3


def test_unknown_key_is_rejected(tmp_path: Path) -> None:
    path = variant(tmp_path, {"weekend_factor = 0.4": "weekend_factor = 0.4\nweekend_factr = 0.4"})
    with pytest.raises(ConfigError, match=r"traffic: unknown keys \['weekend_factr'\]"):
        load(path)


def test_missing_key_is_rejected(tmp_path: Path) -> None:
    path = variant(tmp_path, {"weekend_factor = 0.4\n": ""})
    with pytest.raises(ConfigError, match=r"traffic\.weekend_factor: required"):
        load(path)


def test_template_fields_are_closed(tmp_path: Path) -> None:
    path = variant(
        tmp_path, {'template = "upstream timeout after {ms}ms"': 'template = "upstream timeout after {seconds}s"'}
    )
    with pytest.raises(ConfigError, match="seconds"):
        load(path)


def test_more_novel_scenarios_than_templates_is_rejected(tmp_path: Path) -> None:
    path = variant(tmp_path, {"novel_error = 2": "novel_error = 7"})
    with pytest.raises(ConfigError, match="each template is used once"):
        load(path)


def test_novel_template_may_not_repeat_a_background_one(tmp_path: Path) -> None:
    path = variant(
        tmp_path,
        {'"WAL segment checksum mismatch on branch {id}"': '"upstream timeout after {ms}ms"'},
    )
    with pytest.raises(ConfigError, match="novel error is not novel"):
        load(path)


def test_unknown_timezone_is_rejected(tmp_path: Path) -> None:
    path = variant(tmp_path, {'"Asia/Tokyo"': '"Mars/Olympus_Mons"'})
    with pytest.raises(ConfigError, match="unknown timezone"):
        load(path)


def test_diurnal_amplitude_cannot_make_rates_negative(tmp_path: Path) -> None:
    path = variant(tmp_path, {"diurnal_amplitude = 0.6": "diurnal_amplitude = 1.2"})
    with pytest.raises(ConfigError, match=r"diurnal_amplitude: must be <= 0\.99"):
        load(path)


def test_rename_must_name_a_background_template(tmp_path: Path) -> None:
    path = variant(tmp_path, {'from = "upstream timeout after {ms}ms"': 'from = "upstream timed out after {ms}ms"'})
    with pytest.raises(ConfigError, match=r"is not an errors\.background template"):
        load(path)


def test_a_template_is_renamed_at_most_once(tmp_path: Path) -> None:
    path = variant(
        tmp_path, {'from = "rate limit exceeded for key, retry after {n}s"': 'from = "upstream timeout after {ms}ms"'}
    )
    with pytest.raises(ConfigError, match="renamed once"):
        load(path)


def test_a_rename_may_not_reuse_an_existing_text(tmp_path: Path) -> None:
    path = variant(
        tmp_path, {'to = "upstream request timed out after {ms} ms"': 'to = "slow query on branch {id}: {ms}ms"'}
    )
    with pytest.raises(ConfigError, match="novel error is not novel"):
        load(path)


def test_more_renamed_errors_than_renames_is_rejected(tmp_path: Path) -> None:
    path = variant(tmp_path, {"renamed_error = 1": "renamed_error = 5"})
    with pytest.raises(ConfigError, match=r"need as many errors\.renames"):
        load(path)


@pytest.mark.parametrize(
    ("old", "new", "message"),
    [
        ("echo_branching = [0.5, 0.9]", "echo_branching = [0.5, 1.0]", r"echo_branching: must be <= 0\.99"),
        ("storm_branching = [1.3, 2.0]", "storm_branching = [1.0, 2.0]", r"storm_branching: must be >= 1\.01"),
    ],
)
def test_echoes_die_out_and_storms_do_not(tmp_path: Path, old: str, new: str, message: str) -> None:
    with pytest.raises(ConfigError, match=message):
        load(variant(tmp_path, {old: new}))
