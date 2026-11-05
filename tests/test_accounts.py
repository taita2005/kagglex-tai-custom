"""Tests for named multi-account Kaggle credential management."""

import json
import stat
import time
import unicodedata
from pathlib import Path

import pytest

from kagglex import accounts as acc
from kagglex.cli import create_parser, main
from kagglex.client import _apply_account_credentials, _import_kaggle_api_class
from kagglex.history import RunRecord, get_quota_usage, list_runs, record_run

KEY_PHONG = "0123456789abcdef0123456789abcdef"
KEY_TAN = "fedcba9876543210fedcba9876543210"


def _now_str(offset_sec: float = 0) -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime(time.time() + offset_sec))


def _record(kernel_id: str, accel: str, duration: float, account: str | None = None):
    record_run(
        RunRecord(
            slug=kernel_id.split("/")[1],
            kernel_id=kernel_id,
            title="t",
            command="python train.py",
            accelerator=accel,
            submitted_at=_now_str(-60),
            status="complete",
            duration_sec=duration,
            account=account,
        )
    )


@pytest.fixture
def two_accounts():
    acc.add_account("Phong", "phong_kg", key=KEY_PHONG)
    acc.add_account("Tân", "tan_kg", key=KEY_TAN)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


def test_add_and_load_roundtrip(isolated_home: Path, two_accounts) -> None:
    accounts, default = acc.load_accounts()
    assert set(accounts) == {"Phong", "Tân"}
    assert default == "Phong"  # first account becomes default
    assert accounts["Tân"].username == "tan_kg"
    assert accounts["Tân"].key == KEY_TAN
    assert accounts["Tân"].auth_type == "key"

    path = isolated_home / ".kagglex" / "accounts.toml"
    assert stat.S_IMODE(path.stat().st_mode) == 0o600


def test_lookup_is_unicode_and_case_insensitive(two_accounts) -> None:
    accounts, _ = acc.load_accounts()
    decomposed = unicodedata.normalize("NFD", "Tân")  # how macOS may deliver it
    assert acc.find_account(decomposed, accounts).username == "tan_kg"
    assert acc.find_account("phong", accounts).username == "phong_kg"
    with pytest.raises(acc.AccountError, match="Available: Phong, Tân"):
        acc.find_account("Nobody", accounts)


def test_add_rejects_duplicates_and_reserved_names(two_accounts) -> None:
    with pytest.raises(acc.AccountError, match="already exists"):
        acc.add_account("phong", "x", key=KEY_PHONG)
    with pytest.raises(acc.AccountError, match="reserved"):
        acc.add_account("auto", "x", key=KEY_PHONG)
    with pytest.raises(acc.AccountError, match="exactly one"):
        acc.add_account("New", "x")

    acc.add_account("Phong", "phong_new", token="KGAT_abc", overwrite=True)
    accounts, default = acc.load_accounts()
    assert accounts["Phong"].username == "phong_new"
    assert accounts["Phong"].auth_type == "token"
    assert default == "Phong"


def test_special_characters_survive_toml(isolated_home: Path) -> None:
    acc.add_account('We"ird \\ name', "u", token='tok"en\\x')
    accounts, _ = acc.load_accounts()
    assert accounts['We"ird \\ name'].token == 'tok"en\\x'


def test_use_and_remove(two_accounts) -> None:
    acc.set_default_account("Tân")
    assert acc.load_accounts()[1] == "Tân"

    acc.remove_account("Tân")
    accounts, default = acc.load_accounts()
    assert set(accounts) == {"Phong"}
    assert default == "Phong"  # single remaining account becomes default


# ---------------------------------------------------------------------------
# Quota and selection
# ---------------------------------------------------------------------------


def test_record_run_keeps_reruns_of_same_slug() -> None:
    _record("phong_kg/exp", "t4-2x", 3600)
    _record("tan_kg/exp", "t4-2x", 7200)
    _record("phong_kg/exp", "t4-2x", 1800)
    assert len(list_runs()) == 3
    assert get_quota_usage(username="phong_kg")["gpu"]["used_hours"] == 1.5


def test_quota_is_per_username() -> None:
    _record("phong_kg/a", "t4-2x", 3 * 3600)
    _record("tan_kg/b", "p100", 1 * 3600)
    _record("tan_kg/c", "v3-8", 2 * 3600)

    assert get_quota_usage(username="phong_kg")["gpu"]["used_hours"] == 3.0
    tan = get_quota_usage(username="TAN_KG")
    assert tan["gpu"]["used_hours"] == 1.0
    assert tan["tpu"]["used_hours"] == 2.0
    assert get_quota_usage()["gpu"]["used_hours"] == 4.0  # unfiltered = all


def test_auto_picks_most_remaining(two_accounts) -> None:
    _record("phong_kg/a", "t4-2x", 10 * 3600)
    _record("tan_kg/b", "t4-2x", 2 * 3600)
    _record("tan_kg/c", "v3-8", 15 * 3600)

    assert acc.resolve_account("auto", accelerator="t4-2x").name == "Tân"
    assert acc.resolve_account("AUTO", accelerator="v3-8").name == "Phong"


def test_auto_tie_prefers_default(two_accounts) -> None:
    acc.set_default_account("Tân")
    assert acc.resolve_account("auto").name == "Tân"


def test_resolve_none_uses_default_or_kaggle_fallback(two_accounts) -> None:
    assert acc.resolve_account(None).name == "Phong"
    acc.remove_account("Phong")
    acc.remove_account("Tân")
    assert acc.resolve_account(None) is None
    with pytest.raises(acc.AccountError, match="at least one account"):
        acc.resolve_account("auto")


def test_resolve_account_for_kernel(two_accounts) -> None:
    _record("tan_kg/my-exp", "t4-2x", 60, account="Tân")

    acct, known = acc.resolve_account_for_kernel("my-exp")
    assert acct.name == "Tân" and known == "tan_kg/my-exp"

    acct, _ = acc.resolve_account_for_kernel("phong_kg/other")
    assert acct.name == "Phong"

    acct, _ = acc.resolve_account_for_kernel("my-exp", explicit="Phong")
    assert acct.name == "Phong"

    acct, _ = acc.resolve_account_for_kernel("unknown-slug")
    assert acct.name == "Phong"  # default

    with pytest.raises(acc.AccountError, match="run/push"):
        acc.resolve_account_for_kernel("my-exp", explicit="auto")


# ---------------------------------------------------------------------------
# Credential injection into the real kaggle client (no network)
# ---------------------------------------------------------------------------


def _session_auth(api):
    client = api.build_kaggle_client()
    with client:
        return client._http_client._session.auth


def test_legacy_key_wins_over_stray_access_token(
    monkeypatch: pytest.MonkeyPatch, isolated_home: Path
) -> None:
    api_cls = _import_kaggle_api_class()  # import before setting token env
    monkeypatch.setenv("KAGGLE_API_TOKEN", "someone-elses-token")
    (isolated_home / ".kaggle").mkdir(exist_ok=True)
    (isolated_home / ".kaggle" / "access_token").write_text("another-token")

    api = api_cls()
    _apply_account_credentials(api, acc.Account("Phong", "phong_kg", key=KEY_PHONG))

    assert api.get_config_value("username") == "phong_kg"
    assert _session_auth(api) == ("phong_kg", KEY_PHONG)


def test_token_account_uses_bearer(monkeypatch: pytest.MonkeyPatch) -> None:
    api_cls = _import_kaggle_api_class()
    monkeypatch.setenv("KAGGLE_API_TOKEN", "someone-elses-token")

    api = api_cls()
    _apply_account_credentials(api, acc.Account("Tân", "tan_kg", token="KGAT_mine"))

    assert api.get_config_value("username") == "tan_kg"
    assert _session_auth(api).token == "KGAT_mine"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def test_cli_account_lifecycle(capsys) -> None:
    assert (
        main(["account", "add", "Phong", "--username", "phong_kg", "--key", KEY_PHONG])
        == 0
    )
    assert (
        main(["account", "add", "Tân", "--username", "tan_kg", "--token", "KGAT_x"])
        == 0
    )
    _record("tan_kg/b", "t4-2x", 3 * 3600)
    capsys.readouterr()

    assert main(["list", "--credit"]) == 0
    out = capsys.readouterr().out
    assert "Phong" in out and "Tân" in out
    assert "27.00/30h" in out  # Tân: 30 - 3 hours left
    assert KEY_PHONG not in out and "KGAT_x" not in out

    assert main(["account", "use", "Tân"]) == 0
    assert acc.load_accounts()[1] == "Tân"
    assert main(["account", "remove", "Phong"]) == 0
    assert set(acc.load_accounts()[0]) == {"Tân"}


def test_cli_account_add_from_kaggle_json(tmp_path: Path) -> None:
    kjson = tmp_path / "kaggle.json"
    kjson.write_text(json.dumps({"username": "phong_kg", "key": KEY_PHONG}))
    assert main(["account", "add", "Phong", "--json", str(kjson)]) == 0
    assert acc.load_accounts()[0]["Phong"].key == KEY_PHONG


def test_cli_account_add_prompt_detects_secret_type(monkeypatch) -> None:
    monkeypatch.setattr("kagglex.cli._read_secret", lambda prompt: KEY_PHONG)
    assert main(["account", "add", "Phong", "--username", "phong_kg"]) == 0
    monkeypatch.setattr("kagglex.cli._read_secret", lambda prompt: "KGAT_new")
    assert main(["account", "add", "Tân", "--username", "tan_kg"]) == 0
    accounts, _ = acc.load_accounts()
    assert accounts["Phong"].auth_type == "key"
    assert accounts["Tân"].auth_type == "token"


def test_cli_unknown_account_is_clean_error(caplog, sample_script: Path) -> None:
    rc = main(["run", "--file", str(sample_script), "--account", "Ghost", "--dry-run"])
    assert rc == 1
    assert "Unknown Kaggle account 'Ghost'" in caplog.text
    assert "Traceback" not in caplog.text


def test_cli_run_auto_account_activates_and_records(
    two_accounts, sample_script: Path, mock_kaggle_api
) -> None:
    _record("phong_kg/a", "t4-2x", 20 * 3600)
    rc = main(
        [
            "run",
            "--file",
            str(sample_script),
            "--dir",
            str(sample_script.parent),
            "--account",
            "auto",
            "--dry-run",
        ]
    )
    assert rc == 0
    assert acc.get_active_account().name == "Tân"


def test_cli_status_uses_owning_account(two_accounts, mock_kaggle_api) -> None:
    _record("tan_kg/my-exp", "t4-2x", 60, account="Tân")
    assert main(["status", "my-exp"]) == 0
    assert acc.get_active_account().name == "Tân"
    mock_kaggle_api.kernels_status.assert_called_with("tan_kg/my-exp")


def test_help_lists_all_commands_and_flags() -> None:
    text = create_parser().format_help()
    for needle in (
        "kagglex account add",
        "kagglex account list",
        "--credit",
        "--account",
        "--gpu",
    ):
        assert needle in text


def test_cli_account_test_reports_each_account(
    two_accounts, monkeypatch, capsys
) -> None:
    results = {"phong_kg": (True, "connected"), "tan_kg": (False, "rejected")}
    monkeypatch.setattr("kagglex.cli.verify_account", lambda a: results[a.username])
    assert main(["account", "test"]) == 1
    out = capsys.readouterr().out
    assert "[OK  ] Phong: connected" in out
    assert "[FAIL] Tân: rejected" in out
    assert main(["account", "test", "Phong"]) == 0


def test_verify_account_detects_token_owner_mismatch(monkeypatch) -> None:
    from unittest.mock import MagicMock

    from kagglex.client import verify_account

    api = MagicMock()
    api._introspect_token.return_value = "someone_else"
    monkeypatch.setattr("kagglex.client.get_kaggle_api", lambda: api)
    ok, msg = verify_account(acc.Account("Tân", "tan_kg", token="KGAT_x"))
    assert not ok and "someone_else" in msg
    assert acc.get_active_account() is None  # previous active account restored
