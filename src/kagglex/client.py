"""Kaggle API client wrapper for pre-flight checks, kernel dispatch, monitoring, and output sync."""

import contextlib
import fnmatch
import io
import json
import logging
import shutil
import sys
import tempfile
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

DEFAULT_OUTPUT_EXCLUDES = [
    "cache/**",
    "cache",
    "pkg_src/**",
    "pkg_src",
    "pkg_payload.zip",
    "data_payload.zip",
    "__results__.html",
    "*.pyc",
    "*.pt",  # Unless explicitly included, skip raw PyTorch weights in default pull
]


def _import_kaggle_api_class() -> Any:
    # Importing the `kaggle` package runs an authentication attempt at import
    # time that prints a long help banner when no default credentials exist
    # (normal when only kagglex named accounts are configured). Silence it.
    with contextlib.redirect_stdout(io.StringIO()):
        from kaggle.api.kaggle_api_extended import (
            KaggleApi,  # type: ignore[import-untyped]
        )
    return KaggleApi


def _apply_account_credentials(api: Any, account: Any) -> None:
    """Authenticate `api` with a kagglex named account, bypassing Kaggle's own lookup.

    Kaggle's lookup prefers KAGGLE_API_TOKEN / ~/.kaggle/access_token over any
    username+key, which would silently run jobs on the wrong account.
    """
    try:
        values = dict(api.read_config_file(quiet=True) or {})
    except Exception:
        values = {}
    # Keep non-credential settings (proxy, path, ...) but drop every credential.
    for k in ("username", "key", "token", "api_token", "auth_method"):
        values.pop(k, None)

    values["username"] = account.username
    if account.token:
        values["token"] = account.token
    else:
        values["key"] = account.key

    try:
        from kaggle.api.kaggle_api_extended import (
            AuthMethod,  # type: ignore[import-untyped]
        )

        method = AuthMethod.ACCESS_TOKEN if account.token else AuthMethod.LEGACY_API_KEY
        values["auth_method"] = str(method)
    except ImportError:
        pass  # older kaggle releases have no AuthMethod

    api.config_values = values
    api._authenticated = True

    if account.token:
        return

    # kagglesdk's HTTP client also consults KAGGLE_API_TOKEN / ~/.kaggle/access_token
    # before username+key. Pin the session auth to this account's key instead.
    original_build = getattr(api, "build_kaggle_client", None)
    if original_build is None:
        return

    def build_pinned_client() -> Any:
        client = original_build()
        http = getattr(client, "_http_client", None)
        if http is not None and hasattr(http, "_try_fill_auth"):

            def fill_auth() -> None:
                http._session.auth = (account.username, account.key)
                http._signed_in = True

            http._try_fill_auth = fill_auth
        return client

    api.build_kaggle_client = build_pinned_client


def get_kaggle_api() -> Any:
    """Instantiate and authenticate a KaggleApi client.

    Uses the active kagglex named account if one is set (see kagglex.accounts),
    otherwise Kaggle's default credential lookup (~/.kaggle/kaggle.json, env vars).
    """
    from kagglex.accounts import get_active_account

    account = get_active_account()
    try:
        api = _import_kaggle_api_class()()
        if account is not None:
            _apply_account_credentials(api, account)
        else:
            api.authenticate()
        return api
    except (Exception, SystemExit) as e:
        # kaggle's authenticate() calls exit(1) when no credentials are found.
        logger.error("Failed to authenticate Kaggle API: %s", e)
        raise RuntimeError(
            f"Kaggle API authentication failed ({type(e).__name__}: {e}). Add an account with "
            "`kagglex account add <name> --username <kaggle-username>`, or ensure "
            "~/.kaggle/kaggle.json exists or KAGGLE_USERNAME and KAGGLE_KEY are set."
        ) from e


def verify_account(account: Any) -> tuple[bool, str]:
    """Contact Kaggle with a named account's credentials.

    For token accounts, also checks the token really belongs to the stored
    username (kernel ids are built from it). Returns (ok, message).
    """
    from kagglex.accounts import get_active_account, set_active_account

    previous = get_active_account()
    set_active_account(account)
    try:
        api = get_kaggle_api()
        if account.token and hasattr(api, "_introspect_token"):
            owner = api._introspect_token(account.token)
            if not owner:
                return False, "token is invalid, expired or revoked"
            if owner.casefold() != account.username.casefold():
                return False, (
                    f"token belongs to Kaggle user '{owner}', not '{account.username}' "
                    "(fix with: kagglex account add ... --force)"
                )
        api.kernels_list(mine=True, page_size=1)
        return True, f"connected as Kaggle user '{account.username}'"
    except Exception as e:
        status = getattr(getattr(e, "response", None), "status_code", None)
        if status in (401, 403):
            return False, f"Kaggle rejected the credentials (HTTP {status})"
        return False, f"{type(e).__name__}: {e}"
    finally:
        set_active_account(previous)


def check_kaggle_health() -> tuple[bool, str]:
    """Verify Kaggle credentials and network reachability.

    Returns:
        Tuple of (is_healthy, username_or_error).
    """
    try:
        user = get_authenticated_username()
        return True, user
    except Exception as e:
        return False, str(e)


def get_authenticated_username() -> str:
    """Retrieve the current authenticated Kaggle username."""
    api = get_kaggle_api()
    username = api.get_config_value("username")
    if (
        not username
        and hasattr(api, "config_values")
        and "username" in api.config_values
    ):
        username = api.config_values["username"]
    if not username:
        raise ValueError("Could not determine authenticated Kaggle username.")
    return str(username)


def verify_dataset_sources(dataset_slugs: list[str]) -> list[str]:
    """Check existence of requested dataset sources on Kaggle.

    Returns:
        List of missing or inaccessible dataset slugs.
    """
    if not dataset_slugs:
        return []

    api = get_kaggle_api()
    missing = []
    for slug in dataset_slugs:
        slug = slug.strip()
        try:
            status = api.dataset_status(slug)
            if not status:
                missing.append(slug)
        except Exception:
            missing.append(slug)
    return missing


def push_kernel(staging_dir: Path) -> tuple[str, str]:
    """Push staged kernel directory to Kaggle.

    Args:
        staging_dir: Path containing kernel-metadata.json and code.

    Returns:
        Tuple of (actual_kernel_id, kernel_url).
    """
    metadata_file = staging_dir / "kernel-metadata.json"
    if not metadata_file.exists():
        raise FileNotFoundError(f"kernel-metadata.json not found in {staging_dir}")

    with open(metadata_file, encoding="utf-8") as f:
        meta = json.load(f)
    fallback_id = meta.get("id", "")

    logger.info("Submitting kernel '%s' to Kaggle...", fallback_id)
    api = get_kaggle_api()
    response = api.kernels_push(str(staging_dir))

    actual_id = fallback_id
    url = f"https://www.kaggle.com/code/{fallback_id}"

    if hasattr(response, "url") and response.url:
        url = response.url
        if "/code/" in url:
            actual_id = url.split("/code/")[-1].strip("/")

    logger.info("Kernel successfully submitted: %s (id: %s)", url, actual_id)
    if hasattr(response, "error") and response.error:
        logger.warning("Kernel push note: %s", response.error)
    return actual_id, url


def cancel_kernel(kernel_id: str) -> bool:
    """Cancel a running or queued Kaggle kernel.

    Args:
        kernel_id: Identifier of the kernel (e.g. 'username/slug').

    Returns:
        True if cancellation was signaled, False otherwise.
    """
    api = get_kaggle_api()
    logger.info("Requesting cancellation for kernel '%s'...", kernel_id)
    try:
        if hasattr(api, "kernels_cancel"):
            api.kernels_cancel(kernel_id)
            logger.info("Cancellation request sent for '%s'.", kernel_id)
            return True

        # Attempt cancellation via kagglesdk if available
        if hasattr(api, "build_kaggle_client"):
            try:
                from kagglesdk.kernels.types.kernels_api_service import (  # type: ignore[import-untyped]
                    ApiCancelKernelSessionRequest,
                )

                with api.build_kaggle_client() as k:
                    req = ApiCancelKernelSessionRequest()
                    k.kernels.kernels_api_client.cancel_kernel_session(req)
                    return True
            except Exception as rpc_err:
                logger.debug("RPC session cancel attempt note: %s", rpc_err)

        logger.warning(
            "Kaggle's public API does not support programmatic session cancellation. "
            "To terminate this kernel, click 'Cancel' / 'Stop' in the Kaggle web interface: "
            "https://www.kaggle.com/code/%s",
            kernel_id,
        )
        return False
    except Exception as e:
        logger.error(
            "Failed to cancel kernel '%s': %s. To cancel manually: https://www.kaggle.com/code/%s",
            kernel_id,
            e,
            kernel_id,
        )
        return False


def get_kernel_status(kernel_id: str) -> dict[str, Any]:
    """Fetch execution status for a Kaggle kernel.

    Args:
        kernel_id: Kernel identifier (e.g. 'username/slug').

    Returns:
        Dictionary containing status and message info.
    """
    api = get_kaggle_api()
    status_obj = api.kernels_status(kernel_id)

    status_str = "unknown"
    failure_msg = None

    if hasattr(status_obj, "status"):
        val = status_obj.status
        status_str = getattr(val, "name", str(val)).lower()
        failure_msg = getattr(status_obj, "failure_message", None)
    elif isinstance(status_obj, dict):
        status_str = str(status_obj.get("status", "unknown")).lower()
        failure_msg = status_obj.get("failureMessage")
    else:
        status_str = str(status_obj).lower()

    if "kernelworkerstatus." in status_str:
        status_str = status_str.split("kernelworkerstatus.")[-1]

    return {
        "id": kernel_id,
        "status": status_str,
        "failure_message": failure_msg,
    }


def poll_kernel(
    kernel_id: str,
    poll_interval_sec: int = 20,
    timeout_sec: int = 43200,
    on_status_change: Callable[[str, float], None] | None = None,
) -> dict[str, Any]:
    """Poll kernel status until completion, failure, or timeout.

    Args:
        kernel_id: Identifier of the kernel to monitor.
        poll_interval_sec: Seconds between status checks.
        timeout_sec: Maximum seconds to wait before timeout.
        on_status_change: Optional callback invoked when status changes.

    Returns:
        Final status dictionary.
    """
    logger.info("Monitoring Kaggle kernel '%s'...", kernel_id)
    start_time = time.time()
    last_status = ""

    terminal_statuses = {"complete", "error", "cancelacknowledged", "failed"}

    while True:
        elapsed = time.time() - start_time
        if elapsed > timeout_sec:
            raise TimeoutError(
                f"Kernel {kernel_id} timed out after {elapsed:.0f} seconds."
            )

        try:
            status_info = get_kernel_status(kernel_id)
            current_status = status_info["status"]
        except Exception as e:
            logger.warning(
                "Transient error checking kernel status (%s). Retrying in %ds...",
                e,
                poll_interval_sec,
            )
            time.sleep(poll_interval_sec)
            continue

        if current_status != last_status:
            logger.info(
                "Kernel status changed: [%s] (elapsed: %.0fs)",
                current_status.upper(),
                elapsed,
            )
            if on_status_change:
                on_status_change(current_status, elapsed)
            last_status = current_status

        if current_status in terminal_statuses:
            logger.info("Kernel reached terminal status: %s", current_status.upper())
            status_info["duration_sec"] = elapsed
            return status_info

        time.sleep(poll_interval_sec)


def stream_kernel_logs(
    kernel_id: str, on_line: Callable[[str], None] | None = None
) -> None:
    """Stream live logs from a running Kaggle kernel.

    Args:
        kernel_id: Identifier of the kernel (e.g. 'username/slug').
        on_line: Optional callback for each streamed log line.
    """
    api = get_kaggle_api()
    logger.info("Connecting to log stream for '%s'...", kernel_id)
    try:
        for chunk in api.kernels_logs_stream(kernel_id):
            data = chunk.get("data", "")
            if on_line:
                on_line(data)
            else:
                sys.stdout.write(data)
                sys.stdout.flush()
    except Exception as e:
        logger.debug("Log streaming completed or unavailable: %s", e)


def _match_pattern(rel_path_str: str, patterns: list[str]) -> bool:
    """Check if path string matches any glob pattern."""
    for pat in patterns:
        if fnmatch.fnmatch(rel_path_str, pat) or fnmatch.fnmatch(
            Path(rel_path_str).name, pat
        ):
            return True
        if pat.endswith("/**") and rel_path_str.startswith(pat[:-3]):
            return True
    return False


def _glob_to_regex(pat: str) -> str:
    """Convert a glob pattern to a regex string for Kaggle's server-side filtering."""
    p = pat.strip()
    if p.endswith("/**"):
        p = p[:-3] + "/*"
    elif p.endswith("/"):
        p = p + "*"
    return fnmatch.translate(p)


def _build_server_fetch_pattern(
    include_patterns: list[str] | None = None,
    exclude_patterns: list[str] | None = None,
) -> str:
    """Build a server-side regex pattern for Kaggle's api.kernels_output().

    Args:
        include_patterns: Optional list of glob patterns to include.
        exclude_patterns: Optional list of glob patterns to exclude.

    Returns:
        A compiled-safe regex string.
    """
    if include_patterns:
        sub_regexes = [_glob_to_regex(p) for p in include_patterns if p.strip()]
        if sub_regexes:
            if len(sub_regexes) == 1:
                return sub_regexes[0]
            return "|".join(f"({r})" for r in sub_regexes)

    # Default fetch pattern: fetch outputs/ while skipping large .pt weights and .pyc bytecode server-side
    return r"(?s:outputs/(?!.*(\.pt|\.pyc)$).*)\Z"


def pull_kernel_output(
    kernel_id: str,
    destination_dir: Path,
    include_patterns: list[str] | None = None,
    exclude_patterns: list[str] | None = None,
) -> list[Path]:
    """Download output artifacts from a completed Kaggle kernel with selective filtering.

    Args:
        kernel_id: Identifier of the kernel (e.g. 'username/slug').
        destination_dir: Directory to save downloaded files.
        include_patterns: Optional glob patterns to include.
        exclude_patterns: Optional glob patterns to exclude.

    Returns:
        List of saved file paths.
    """
    destination_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Downloading outputs for '%s' to %s...", kernel_id, destination_dir)

    effective_excludes = list(exclude_patterns or DEFAULT_OUTPUT_EXCLUDES)
    api = get_kaggle_api()
    server_pattern = _build_server_fetch_pattern(
        include_patterns=include_patterns, exclude_patterns=exclude_patterns
    )

    saved_files: list[Path] = []

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        try:
            api.kernels_output(
                kernel_id,
                path=str(tmp_path),
                file_pattern=server_pattern,
                force=True,
                quiet=True,
            )
        except Exception as e:
            logger.debug("Server fetch pattern '%s' note: %s", server_pattern, e)

        for file_path in tmp_path.rglob("*"):
            if not file_path.is_file():
                continue

            rel_str = str(file_path.relative_to(tmp_path))

            if _match_pattern(rel_str, effective_excludes):
                logger.debug("Excluding output file: %s", rel_str)
                continue

            if include_patterns and not _match_pattern(rel_str, include_patterns):
                logger.debug("Skipping non-included output file: %s", rel_str)
                continue

            clean_rel = rel_str
            if destination_dir.name == "outputs" and rel_str.startswith("outputs/"):
                clean_rel = rel_str[len("outputs/") :]

            dest_file = destination_dir / clean_rel
            dest_file.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(file_path, dest_file)
            saved_files.append(dest_file)

    logger.info("Downloaded %d output files to %s.", len(saved_files), destination_dir)
    return saved_files


def append_experiment_record(
    record_file: Path,
    kernel_id: str,
    status_info: dict[str, Any],
    output_dir: Path | None = None,
) -> None:
    """Update or append Kaggle execution details to an experiment record file."""
    if not record_file.exists():
        logger.warning(
            "Experiment record file %s does not exist. Skipping update.", record_file
        )
        return

    kernel_url = f"https://www.kaggle.com/code/{kernel_id}"
    status_str = status_info.get("status", "unknown").upper()
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())

    log_entry = [
        "",
        "### Kaggle Remote Execution",
        "",
        f"- **Kernel ID:** `{kernel_id}`",
        f"- **Kernel URL:** {kernel_url}",
        f"- **Completed At:** {timestamp}",
        f"- **Status:** `{status_str}`",
    ]

    if status_info.get("failure_message"):
        log_entry.append(f"- **Error:** {status_info['failure_message']}")

    if output_dir:
        try:
            rel_output = output_dir.relative_to(record_file.parent)
        except ValueError:
            rel_output = output_dir
        log_entry.append(f"- **Downloaded Outputs:** `{rel_output}`")

    log_entry.append("")

    with open(record_file, "a", encoding="utf-8") as f:
        f.write("\n".join(log_entry))

    logger.info("Updated experiment record: %s", record_file)
