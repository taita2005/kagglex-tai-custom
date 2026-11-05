"""Named Kaggle account registry stored in ~/.kagglex/accounts.toml.

Each account maps a human-friendly name (e.g. "Phong") to a Kaggle username
plus either a legacy API key (from kaggle.json) or a new-style API token.
The process-wide *active* account is consumed by ``client.get_kaggle_api``.
"""

import json
import logging
import os
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path

from kagglex.config import read_toml_file

logger = logging.getLogger(__name__)

AUTO_ACCOUNT = "auto"
ACCOUNTS_FILENAME = "accounts.toml"

# Legacy Kaggle API keys (kaggle.json "key") are 32 lowercase hex characters.
_LEGACY_KEY_RE = re.compile(r"^[0-9a-f]{32}$")
_TPU_ACCELERATORS = {"v3-8", "v2-8", "tpu"}


class AccountError(ValueError):
    """Unknown account or invalid account operation (shown to users without a traceback)."""


@dataclass
class Account:
    """A named Kaggle credential set."""

    name: str
    username: str
    key: str | None = None
    token: str | None = None

    @property
    def auth_type(self) -> str:
        """'token' for new-style API tokens, 'key' for legacy kaggle.json keys."""
        return "token" if self.token else "key"


def normalize_name(name: str) -> str:
    """Normalize an account name (NFC, so 'Tân' typed on macOS matches stored 'Tân')."""
    return unicodedata.normalize("NFC", name.strip())


def looks_like_legacy_key(secret: str) -> bool:
    """Whether a secret has the shape of a legacy kaggle.json API key."""
    return bool(_LEGACY_KEY_RE.match(secret.strip()))


def get_accounts_file(path: Path | None = None) -> Path:
    """Path to the accounts registry file."""
    return path or (Path.home() / ".kagglex" / ACCOUNTS_FILENAME)


def load_accounts(
    path: Path | None = None,
) -> tuple[dict[str, Account], str | None]:
    """Load all accounts and the default account name."""
    data = read_toml_file(get_accounts_file(path))
    accounts: dict[str, Account] = {}

    raw_accounts = data.get("accounts") or {}
    if isinstance(raw_accounts, dict):
        for raw_name, entry in raw_accounts.items():
            if not isinstance(entry, dict) or not entry.get("username"):
                logger.warning("Skipping malformed account entry '%s'.", raw_name)
                continue
            if not entry.get("key") and not entry.get("token"):
                logger.warning("Account '%s' has no key or token; skipping.", raw_name)
                continue
            name = normalize_name(str(raw_name))
            accounts[name] = Account(
                name=name,
                username=str(entry["username"]),
                key=str(entry["key"]) if entry.get("key") else None,
                token=str(entry["token"]) if entry.get("token") else None,
            )

    default = data.get("default")
    default = normalize_name(default) if isinstance(default, str) else None
    if default not in accounts:
        default = None
    return accounts, default


def _toml_str(value: str) -> str:
    # JSON string escapes (\" \\ \n \uXXXX ...) are a valid subset of TOML basic strings.
    return json.dumps(value, ensure_ascii=False)


def save_accounts(
    accounts: dict[str, Account],
    default: str | None,
    path: Path | None = None,
) -> Path:
    """Write the registry atomically with owner-only permissions (0600)."""
    target = get_accounts_file(path)
    target.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        "# kagglex Kaggle accounts. Contains secrets: keep this file private (chmod 600).",
        "# Manage with: kagglex account add | list | use | remove",
        "",
    ]
    if default:
        lines += [f"default = {_toml_str(default)}", ""]
    for name in sorted(accounts, key=str.casefold):
        acct = accounts[name]
        lines.append(f"[accounts.{_toml_str(name)}]")
        lines.append(f"username = {_toml_str(acct.username)}")
        if acct.token:
            lines.append(f"token = {_toml_str(acct.token)}")
        else:
            lines.append(f"key = {_toml_str(acct.key or '')}")
        lines.append("")

    tmp = target.with_name(target.name + ".tmp")
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    os.chmod(tmp, 0o600)
    os.replace(tmp, target)
    return target


def _validate_name(name: str) -> str:
    name = normalize_name(name)
    if not name:
        raise AccountError("Account name cannot be empty.")
    if name.casefold() == AUTO_ACCOUNT:
        raise AccountError(
            f"'{AUTO_ACCOUNT}' is reserved and cannot be an account name."
        )
    if any(unicodedata.category(c).startswith("C") for c in name):
        raise AccountError("Account name cannot contain control characters.")
    return name


def find_account(name: str, accounts: dict[str, Account]) -> Account:
    """Look up an account by name (exact, then case-insensitive)."""
    wanted = normalize_name(name)
    if wanted in accounts:
        return accounts[wanted]
    matches = [a for n, a in accounts.items() if n.casefold() == wanted.casefold()]
    if len(matches) == 1:
        return matches[0]
    available = ", ".join(sorted(accounts, key=str.casefold)) or "(none)"
    raise AccountError(
        f"Unknown Kaggle account '{name}'. Available: {available}. "
        "Add one with: kagglex account add <name> --username <kaggle-username>"
    )


def find_account_by_username(
    username: str, accounts: dict[str, Account]
) -> Account | None:
    """Find the account that owns a Kaggle username (case-insensitive)."""
    for acct in accounts.values():
        if acct.username.casefold() == username.casefold():
            return acct
    return None


def add_account(
    name: str,
    username: str,
    key: str | None = None,
    token: str | None = None,
    make_default: bool = False,
    overwrite: bool = False,
    path: Path | None = None,
) -> Account:
    """Register a named account. The first account becomes the default."""
    name = _validate_name(name)
    username = username.strip()
    if not username:
        raise AccountError("Kaggle username cannot be empty.")
    key = key.strip() if key else None
    token = token.strip() if token else None
    if bool(key) == bool(token):
        raise AccountError("Provide exactly one of an API key or an API token.")

    accounts, default = load_accounts(path)
    existing = next((n for n in accounts if n.casefold() == name.casefold()), None)
    if existing and not overwrite:
        raise AccountError(
            f"Account '{existing}' already exists. Use --force to replace it."
        )
    if existing:
        del accounts[existing]
        if default == existing:
            default = name

    acct = Account(name=name, username=username, key=key, token=token)
    accounts[name] = acct
    if make_default or default is None:
        default = name
    save_accounts(accounts, default, path)
    return acct


def remove_account(name: str, path: Path | None = None) -> Account:
    """Remove an account. If it was the default and one account remains, that one becomes default."""
    accounts, default = load_accounts(path)
    acct = find_account(name, accounts)
    del accounts[acct.name]
    if default == acct.name:
        default = next(iter(accounts)) if len(accounts) == 1 else None
    save_accounts(accounts, default, path)
    return acct


def set_default_account(name: str, path: Path | None = None) -> Account:
    """Mark an account as the default."""
    accounts, _ = load_accounts(path)
    acct = find_account(name, accounts)
    save_accounts(accounts, acct.name, path)
    return acct


def account_quota(
    acct: Account,
    window_days: int = 7,
    gpu_limit_hours: float = 30.0,
    tpu_limit_hours: float = 20.0,
    history_path: Path | None = None,
) -> dict:
    """Estimated quota usage for one account (from local run history)."""
    from kagglex.history import get_quota_usage

    return get_quota_usage(
        history_path=history_path,
        window_days=window_days,
        gpu_limit_hours=gpu_limit_hours,
        tpu_limit_hours=tpu_limit_hours,
        username=acct.username,
    )


def pick_auto_account(
    accounts: dict[str, Account],
    default: str | None,
    accelerator: str = "t4-2x",
    window_days: int = 7,
    gpu_limit_hours: float = 30.0,
    tpu_limit_hours: float = 20.0,
    history_path: Path | None = None,
) -> Account:
    """Choose the account with the most remaining quota for the accelerator.

    TPU runs compare TPU hours; everything else compares GPU hours.
    Ties prefer the default account, then alphabetical order.
    """
    if not accounts:
        raise AccountError(
            "'--account auto' needs at least one account. "
            "Add one with: kagglex account add <name> --username <kaggle-username>"
        )
    kind = "tpu" if accelerator.lower() in _TPU_ACCELERATORS else "gpu"

    remaining = {
        name: float(
            account_quota(
                acct, window_days, gpu_limit_hours, tpu_limit_hours, history_path
            )[kind]["remaining_hours"]
        )
        for name, acct in accounts.items()
    }
    best = min(
        accounts.values(),
        key=lambda a: (-remaining[a.name], a.name != default, a.name.casefold()),
    )
    best_remaining = remaining[best.name]
    if best_remaining <= 0:
        logger.warning(
            "All accounts appear to have exhausted their estimated %s quota.",
            kind.upper(),
        )
    logger.info(
        "Auto-selected account '%s' (%.2f estimated %s hrs remaining).",
        best.name,
        best_remaining,
        kind.upper(),
    )
    return best


def resolve_account(
    selector: str | None,
    accelerator: str = "t4-2x",
    window_days: int = 7,
    gpu_limit_hours: float = 30.0,
    tpu_limit_hours: float = 20.0,
    path: Path | None = None,
    history_path: Path | None = None,
) -> Account | None:
    """Resolve an account selector.

    - None: the default account, or None when no accounts are registered
      (caller then falls back to Kaggle's own credentials, e.g. ~/.kaggle/kaggle.json).
    - "auto": the account with most remaining estimated quota.
    - anything else: the account with that name.
    """
    accounts, default = load_accounts(path)
    if selector is None or not str(selector).strip():
        return accounts.get(default) if default else None
    if normalize_name(selector).casefold() == AUTO_ACCOUNT:
        return pick_auto_account(
            accounts,
            default,
            accelerator=accelerator,
            window_days=window_days,
            gpu_limit_hours=gpu_limit_hours,
            tpu_limit_hours=tpu_limit_hours,
            history_path=history_path,
        )
    return find_account(selector, accounts)


def resolve_account_for_kernel(
    kernel_ref: str,
    explicit: str | None = None,
    path: Path | None = None,
    history_path: Path | None = None,
) -> tuple[Account | None, str | None]:
    """Pick the account that owns a kernel, for status/logs/pull/cancel.

    Returns (account, full_kernel_id_from_history_or_None). The kernel owner is
    taken from 'user/slug' or, for a bare slug, from the newest matching run
    record; the account registered for that username is used. An explicit
    --account always wins. Unknown owners fall back to Kaggle's own credentials.
    """
    from kagglex.history import get_run, kernel_owner

    accounts, default = load_accounts(path)
    rec = get_run(kernel_ref, history_path)
    known_id = rec.kernel_id if rec else None

    if explicit:
        if normalize_name(explicit).casefold() == AUTO_ACCOUNT:
            raise AccountError("'--account auto' is only supported by run/push.")
        return find_account(explicit, accounts), known_id

    owner = kernel_owner(kernel_ref) or (kernel_owner(known_id) if known_id else "")
    if owner:
        acct = find_account_by_username(owner, accounts)
        if acct is None and accounts:
            logger.warning(
                "No registered account for Kaggle user '%s'; using Kaggle default credentials.",
                owner,
            )
        return acct, known_id
    return (accounts.get(default) if default else None), known_id


# ---------------------------------------------------------------------------
# Process-wide active account (consumed by client.get_kaggle_api)
# ---------------------------------------------------------------------------

_active_account: Account | None = None


def set_active_account(account: Account | None) -> None:
    """Set the account used by subsequent Kaggle API calls (None = Kaggle defaults)."""
    global _active_account
    changed = account != _active_account
    _active_account = account
    if account and changed:
        logger.info("Using Kaggle account '%s' (%s).", account.name, account.username)


def get_active_account() -> Account | None:
    """The account currently used for Kaggle API calls, if any."""
    return _active_account
