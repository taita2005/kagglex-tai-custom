"""Command-line interface for kagglex."""

import argparse
import logging
import os
import sys
from pathlib import Path

from kagglex.accounts import (
    AccountError,
    account_quota,
    add_account,
    find_account,
    get_accounts_file,
    load_accounts,
    looks_like_legacy_key,
    remove_account,
    resolve_account,
    resolve_account_for_kernel,
    set_active_account,
    set_default_account,
)
from kagglex.api import KaggleRunner
from kagglex.client import (
    append_experiment_record,
    cancel_kernel,
    check_kaggle_health,
    get_authenticated_username,
    get_kernel_status,
    pull_kernel_output,
    stream_kernel_logs,
    verify_account,
)
from kagglex.config import DatasetConfig, RunConfig, load_project_config
from kagglex.dataset import create_dataset_metadata, push_dataset
from kagglex.history import get_quota_usage, kernel_owner, list_runs, update_run
from kagglex.workspace import find_repo_root

logger = logging.getLogger("kagglex")


def setup_logging(verbose: bool = False) -> None:
    """Configure logging format and verbosity."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def parse_env_args(env_items: list[str] | None, env_file: str | None) -> dict[str, str]:
    """Parse --env KEY=VAL arguments and optional .env file."""
    env_dict: dict[str, str] = {}

    if env_file:
        env_path = Path(env_file).resolve()
        if env_path.is_file():
            for line in env_path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    env_dict[k.strip()] = v.strip().strip("'\"")

    if env_items:
        for item in env_items:
            if "=" in item:
                k, v = item.split("=", 1)
                env_dict[k.strip()] = v.strip()
            else:
                val = os.environ.get(item, "")
                env_dict[item] = val

    return env_dict


def handle_run(args: argparse.Namespace) -> int:
    """Handle the 'run' command: Stage, dispatch, monitor, and retrieve outputs."""
    project_dir = Path(args.dir).resolve() if args.dir else find_repo_root()
    proj_cfg = load_project_config(project_dir)

    # 1. Environment variables
    file_env = proj_cfg.get("env", {})
    env_vars = parse_env_args(args.env, args.env_file or proj_cfg.get("env_file"))
    if file_env and isinstance(file_env, dict):
        merged_env = {str(k): str(v) for k, v in file_env.items()}
        merged_env.update(env_vars)
        env_vars = merged_env

    # 2. Target file & command
    target_file = (
        Path(args.file).resolve()
        if args.file
        else (
            (project_dir / proj_cfg["file"]).resolve() if "file" in proj_cfg else None
        )
    )
    cmd = args.command or proj_cfg.get("command")
    if not cmd and target_file:
        cmd = f"python {target_file.name}"
    elif not cmd:
        logger.error(
            "Must specify --command or --file (or configure in kagglex.toml / pyproject.toml)"
        )
        return 1

    # 3. Title & Slug
    title = args.title or proj_cfg.get("title")
    if not title:
        if target_file:
            title = f"Run {target_file.stem}"
        else:
            title = "Kaggle Experiment Run"

    slug = args.slug or proj_cfg.get("slug")

    # 4. Accelerator / GPU
    gpu_type = args.gpu
    if gpu_type == "t4-2x" and ("gpu" in proj_cfg or "gpu_type" in proj_cfg):
        gpu_type = str(proj_cfg.get("gpu") or proj_cfg.get("gpu_type"))

    # 5. Flags
    multi_gpu = args.multi_gpu or bool(proj_cfg.get("multi_gpu", False))
    auto_dataset = args.auto_dataset or bool(proj_cfg.get("auto_dataset", False))
    auto_dataset_slug = args.auto_dataset_slug or proj_cfg.get("auto_dataset_slug")
    no_internet = (
        args.no_internet
        or bool(proj_cfg.get("no_internet", False))
        or (proj_cfg.get("enable_internet") is False)
    )

    # 6. Collections
    datasets = (
        args.datasets or proj_cfg.get("datasets") or proj_cfg.get("dataset_slugs") or []
    )
    include_data = args.include_data or proj_cfg.get("include_data") or []
    extra_deps = (
        args.extra_deps
        or proj_cfg.get("extra_deps")
        or proj_cfg.get("extra_pip_deps")
        or []
    )
    kaggle_secrets = args.kaggle_secrets or proj_cfg.get("kaggle_secrets") or []
    parent_kernels = args.parent_kernels or proj_cfg.get("parent_kernels") or []
    include_outputs = args.include_outputs or proj_cfg.get("include_outputs") or []
    exclude_outputs = args.exclude_outputs or proj_cfg.get("exclude_outputs") or []

    # 7. Outputs, records & poll interval
    output_dir_val = args.output_dir or proj_cfg.get("output_dir")
    output_dir = Path(output_dir_val).resolve() if output_dir_val else None

    record_file_val = args.record_file or proj_cfg.get("record_file")
    record_file = Path(record_file_val).resolve() if record_file_val else None

    poll_interval = args.poll_interval
    if poll_interval == 20 and "poll_interval" in proj_cfg:
        poll_interval = int(proj_cfg["poll_interval"])

    # 8. Kaggle account: --account > [tool.kagglex] account > default account > kaggle.json
    gpu_limit, tpu_limit = _quota_limits(proj_cfg)
    acct = resolve_account(
        args.account or proj_cfg.get("account"),
        accelerator=gpu_type,
        gpu_limit_hours=gpu_limit,
        tpu_limit_hours=tpu_limit,
    )
    set_active_account(acct)

    config = RunConfig(
        command=cmd,
        title=title,
        slug=slug,
        project_dir=project_dir,
        target_file=target_file,
        gpu_type=gpu_type,
        multi_gpu=multi_gpu,
        enable_internet=not no_internet,
        dataset_slugs=datasets,
        include_data=include_data,
        extra_pip_deps=extra_deps,
        env_vars=env_vars,
        kaggle_secrets=kaggle_secrets,
        parent_kernels=parent_kernels,
        output_dir=output_dir,
        record_file=record_file,
        include_outputs=include_outputs,
        exclude_outputs=exclude_outputs,
        auto_dataset=auto_dataset,
        auto_dataset_slug=auto_dataset_slug,
        poll_interval=poll_interval,
        account=acct.name if acct else None,
    )

    # Check estimated quota warning (per account when one is selected)
    try:
        usage = get_quota_usage(
            window_days=7,
            gpu_limit_hours=gpu_limit,
            tpu_limit_hours=tpu_limit,
            username=acct.username if acct else None,
        )
        if config.gpu_type in {"t4-2x", "p100"}:
            gpu_info = usage["gpu"]
            if gpu_info["used_pct"] >= 90.0 or gpu_info["remaining_hours"] < 2.0:
                logger.warning(
                    "Estimated GPU quota is %.1f%% used (%.2f / %.1f hrs). Remaining: %.2f hrs.",
                    gpu_info["used_pct"],
                    gpu_info["used_hours"],
                    gpu_info["limit_hours"],
                    gpu_info["remaining_hours"],
                )
        elif config.gpu_type == "v3-8" or config.enable_tpu:
            tpu_info = usage["tpu"]
            if tpu_info["used_pct"] >= 90.0 or tpu_info["remaining_hours"] < 2.0:
                logger.warning(
                    "Estimated TPU quota is %.1f%% used (%.2f / %.1f hrs). Remaining: %.2f hrs.",
                    tpu_info["used_pct"],
                    tpu_info["used_hours"],
                    tpu_info["limit_hours"],
                    tpu_info["remaining_hours"],
                )
    except Exception as e:
        logger.debug("Quota pre-flight check skipped: %s", e)

    runner = KaggleRunner(repo_root=project_dir)

    if args.dry_run:
        logger.info("Dry run requested. Staging bundle locally...")
        staging_dir = runner.stage(config)
        logger.info("Staged bundle successfully at: %s", staging_dir)
        return 0

    # Pre-flight check
    healthy, user_or_err = check_kaggle_health()
    if not healthy:
        logger.error("Kaggle authentication / pre-flight check failed: %s", user_or_err)
        return 1

    job = runner.run(
        config=config,
        wait=not args.no_wait,
        stream=args.stream,
        pull=bool(config.output_dir or config.record_file),
    )

    if args.no_wait:
        logger.info(
            "Kernel submitted. Monitor later with: kagglex status %s", job.kernel_id
        )
        return 0

    status = job.status
    if config.record_file and job.last_status_info:
        append_experiment_record(
            record_file=config.record_file,
            kernel_id=job.kernel_id,
            status_info=job.last_status_info,
            output_dir=config.output_dir,
        )

    return 0 if status == "complete" else 1


def handle_push(args: argparse.Namespace) -> int:
    """Handle 'push' subcommand: stage and submit without waiting."""
    args.no_wait = True
    args.stream = False
    return handle_run(args)


def _resolve_kernel(args: argparse.Namespace) -> str:
    """Activate the account owning `args.kernel` and return the full 'user/slug' id."""
    acct, known_id = resolve_account_for_kernel(
        args.kernel, explicit=getattr(args, "account", None)
    )
    set_active_account(acct)

    kernel_id = args.kernel
    if "/" in kernel_id:
        return kernel_id
    if known_id and (
        acct is None or kernel_owner(known_id).casefold() == acct.username.casefold()
    ):
        return known_id
    return f"{get_authenticated_username()}/{kernel_id}"


def handle_status(args: argparse.Namespace) -> int:
    """Handle 'status' subcommand."""
    kernel_id = _resolve_kernel(args)

    status_info = get_kernel_status(kernel_id)
    logger.info("Kernel '%s' status: %s", kernel_id, status_info["status"].upper())
    if status_info.get("failure_message"):
        logger.warning("Failure details: %s", status_info["failure_message"])

    # Update local history if present
    update_run(
        kernel_id,
        {
            "status": status_info["status"],
            "error_message": status_info.get("failure_message"),
        },
    )
    return 0


def handle_logs(args: argparse.Namespace) -> int:
    """Handle 'logs' subcommand."""
    kernel_id = _resolve_kernel(args)

    stream_kernel_logs(kernel_id)
    return 0


def handle_cancel(args: argparse.Namespace) -> int:
    """Handle 'cancel' subcommand."""
    kernel_id = _resolve_kernel(args)

    ok = cancel_kernel(kernel_id)
    if ok:
        update_run(kernel_id, {"status": "cancelled"})
        logger.info("Job '%s' cancelled.", kernel_id)
        return 0
    return 1


def handle_pull(args: argparse.Namespace) -> int:
    """Handle 'pull' subcommand."""
    kernel_id = _resolve_kernel(args)

    output_dir = Path(args.output_dir)
    pull_kernel_output(
        kernel_id,
        destination_dir=output_dir,
        include_patterns=args.include_outputs or [],
        exclude_patterns=args.exclude_outputs or [],
    )
    update_run(kernel_id, {"output_dir": str(output_dir)})
    return 0


def _print_table(headers: list[str], rows: list[list[str]]) -> None:
    """Write a left-aligned plain-text table to stdout."""
    col_widths = [len(h) for h in headers]
    for row in rows:
        for i, val in enumerate(row):
            col_widths[i] = max(col_widths[i], len(val))

    fmt = "  ".join(f"{{:<{w}}}" for w in col_widths)
    sys.stdout.write(fmt.format(*headers).rstrip() + "\n")
    sys.stdout.write("  ".join("-" * w for w in col_widths) + "\n")
    for row in rows:
        sys.stdout.write(fmt.format(*row).rstrip() + "\n")


def _quota_limits(proj_cfg: dict) -> tuple[float, float]:
    """Weekly GPU/TPU hour limits from config (defaults 30 / 20)."""
    return (
        float(proj_cfg.get("gpu_weekly_limit_hours", 30.0)),
        float(proj_cfg.get("tpu_weekly_limit_hours", 20.0)),
    )


def handle_list(args: argparse.Namespace) -> int:
    """Handle 'list' subcommand (or 'list --credit' for the account overview)."""
    if args.credit:
        return handle_account_list(args)

    runs = list_runs(limit=args.limit)

    if not runs:
        logger.info("No recorded runs found.")
        return 0

    headers = [
        "SLUG / KERNEL ID",
        "ACCOUNT",
        "STATUS",
        "ACCELERATOR",
        "SUBMITTED AT",
        "DURATION",
    ]
    rows = []
    for r in runs:
        dur = f"{r.duration_sec:.1f}s" if r.duration_sec else "-"
        rows.append(
            [
                r.kernel_id,
                r.account or "-",
                r.status.upper(),
                r.accelerator,
                r.submitted_at,
                dur,
            ]
        )

    sys.stdout.write("\n")
    _print_table(headers, rows)
    sys.stdout.write("\n")
    return 0


# ---------------------------------------------------------------------------
# 'account' subcommands
# ---------------------------------------------------------------------------


def _read_secret(prompt: str) -> str:
    import getpass

    if not sys.stdin.isatty():
        return sys.stdin.readline().strip()
    return getpass.getpass(prompt).strip()


def handle_account_add(args: argparse.Namespace) -> int:
    """Handle 'account add': register a named Kaggle account."""
    username = args.username
    key, token = args.key, args.token

    if args.json:
        import json

        json_path = Path(args.json).expanduser()
        try:
            creds = json.loads(json_path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as e:
            logger.error("Could not read kaggle.json file %s: %s", json_path, e)
            return 1
        username = username or creds.get("username")
        key = creds.get("key")
        if not key:
            logger.error("%s has no 'key' field.", json_path)
            return 1

    if not username:
        logger.error("--username is required (or use --json path/to/kaggle.json).")
        return 1

    if not key and not token:
        secret = _read_secret(
            f"Kaggle API key or token for '{username}' (input hidden): "
        )
        if not secret:
            logger.error("No API key or token entered.")
            return 1
        if looks_like_legacy_key(secret):
            key = secret
        else:
            token = secret

    try:
        acct = add_account(
            args.name,
            username=username,
            key=key,
            token=token,
            make_default=args.default,
            overwrite=args.force,
        )
    except ValueError as e:
        logger.error("%s", e)
        return 1

    _, default = load_accounts()
    suffix = " (default)" if default == acct.name else ""
    sys.stdout.write(
        f"Added account '{acct.name}' -> Kaggle user '{acct.username}' "
        f"[{acct.auth_type}]{suffix}.\nSaved to {get_accounts_file()} (chmod 600).\n"
    )
    if args.verify:
        return _test_accounts([acct])
    return 0


def _test_accounts(accts: list) -> int:
    """Check each account against Kaggle and print one result line per account."""
    failed = 0
    for acct in accts:
        ok, message = verify_account(acct)
        failed += not ok
        mark = "OK  " if ok else "FAIL"
        sys.stdout.write(f"[{mark}] {acct.name}: {message}\n")
    return 1 if failed else 0


def handle_account_test(args: argparse.Namespace) -> int:
    """Handle 'account test': verify credentials by contacting Kaggle."""
    accounts, _ = load_accounts()
    if not accounts:
        sys.stdout.write("No Kaggle accounts registered.\n")
        return 1
    if args.names:
        accts = [find_account(n, accounts) for n in args.names]
    else:
        accts = [accounts[n] for n in sorted(accounts, key=str.casefold)]
    return _test_accounts(accts)


def handle_account_list(args: argparse.Namespace) -> int:
    """Handle 'account list' / 'list --credit': accounts with estimated quota."""
    accounts, default = load_accounts()
    if not accounts:
        sys.stdout.write(
            "No Kaggle accounts registered.\n"
            "Add one with: kagglex account add <name> --username <kaggle-username>\n"
        )
        return 0

    proj_cfg = load_project_config(find_repo_root())
    cfg_gpu, cfg_tpu = _quota_limits(proj_cfg)
    days = getattr(args, "days", None) or int(proj_cfg.get("quota_days", 7))
    gpu_limit = getattr(args, "gpu_limit", None) or cfg_gpu
    tpu_limit = getattr(args, "tpu_limit", None) or cfg_tpu

    headers = [
        "",
        "NAME",
        "KAGGLE USER",
        "AUTH",
        "GPU USED",
        "GPU LEFT",
        "TPU USED",
        "TPU LEFT",
        "RUNS",
    ]
    rows = []
    for name in sorted(accounts, key=str.casefold):
        acct = accounts[name]
        usage = account_quota(acct, days, gpu_limit, tpu_limit)
        gpu, tpu = usage["gpu"], usage["tpu"]
        rows.append(
            [
                "*" if name == default else "",
                name,
                acct.username,
                acct.auth_type,
                f"{gpu['used_hours']:.2f}h",
                f"{gpu['remaining_hours']:.2f}/{gpu['limit_hours']:.0f}h",
                f"{tpu['used_hours']:.2f}h",
                f"{tpu['remaining_hours']:.2f}/{tpu['limit_hours']:.0f}h",
                str(len(usage["runs"])),
            ]
        )

    sys.stdout.write(f"\nKaggle accounts - estimated quota, past {days} days:\n\n")
    _print_table(headers, rows)
    sys.stdout.write(
        "\n* = default account. Usage is estimated from runs submitted via kagglex "
        "on this machine\n  (runs finished while kagglex was waiting); Kaggle has "
        "no API for remaining quota.\n\n"
    )
    return 0


def handle_account_use(args: argparse.Namespace) -> int:
    """Handle 'account use': set the default account."""
    try:
        acct = set_default_account(args.name)
    except AccountError as e:
        logger.error("%s", e)
        return 1
    sys.stdout.write(f"Default account is now '{acct.name}' ({acct.username}).\n")
    return 0


def handle_account_remove(args: argparse.Namespace) -> int:
    """Handle 'account remove': delete a stored account."""
    try:
        acct = remove_account(args.name)
    except AccountError as e:
        logger.error("%s", e)
        return 1
    _, default = load_accounts()
    sys.stdout.write(f"Removed account '{acct.name}' ({acct.username}).\n")
    if default:
        sys.stdout.write(f"Default account: '{default}'.\n")
    else:
        sys.stdout.write("No default account set (use: kagglex account use <name>).\n")
    return 0


def handle_quota(args: argparse.Namespace) -> int:
    """Handle 'quota' subcommand to display estimated accelerator usage."""
    repo_root = find_repo_root()
    proj_cfg = load_project_config(repo_root)

    gpu_limit = (
        args.gpu_limit
        if args.gpu_limit != 30.0
        else float(proj_cfg.get("gpu_weekly_limit_hours", 30.0))
    )
    tpu_limit = (
        args.tpu_limit
        if args.tpu_limit != 20.0
        else float(proj_cfg.get("tpu_weekly_limit_hours", 20.0))
    )

    acct = None
    if args.account:
        try:
            acct = find_account(args.account, load_accounts()[0])
        except AccountError as e:
            logger.error("%s", e)
            return 1

    usage = get_quota_usage(
        window_days=args.days,
        gpu_limit_hours=gpu_limit,
        tpu_limit_hours=tpu_limit,
        username=acct.username if acct else None,
    )

    gpu_info = usage["gpu"]
    tpu_info = usage["tpu"]

    scope = f" - account '{acct.name}' ({acct.username})" if acct else ""
    sys.stdout.write(
        f"\nKaggle Accelerator Quota Usage (Past {args.days} Days){scope}:\n\n"
    )

    headers = [
        "ACCELERATOR",
        "USED HOURS",
        "QUOTA LIMIT",
        "REMAINING",
        "UTILIZATION",
        "RUNS",
    ]
    rows = [
        [
            "GPU",
            f"{gpu_info['used_hours']:.2f} hrs",
            f"{gpu_info['limit_hours']:.2f} hrs",
            f"{gpu_info['remaining_hours']:.2f} hrs",
            f"{gpu_info['used_pct']:.1f}%",
            str(gpu_info["run_count"]),
        ],
        [
            "TPU",
            f"{tpu_info['used_hours']:.2f} hrs",
            f"{tpu_info['limit_hours']:.2f} hrs",
            f"{tpu_info['remaining_hours']:.2f} hrs",
            f"{tpu_info['used_pct']:.1f}%",
            str(tpu_info["run_count"]),
        ],
    ]

    col_widths = [len(h) for h in headers]
    for row in rows:
        for i, val in enumerate(row):
            col_widths[i] = max(col_widths[i], len(val))

    fmt = "  ".join(f"{{:<{w}}}" for w in col_widths)
    sys.stdout.write(fmt.format(*headers) + "\n")
    sys.stdout.write("  ".join("-" * w for w in col_widths) + "\n")
    for row in rows:
        sys.stdout.write(fmt.format(*row) + "\n")
    sys.stdout.write("\n")

    runs = usage.get("runs", [])
    if runs:
        sys.stdout.write(f"Recent Runs in Window ({len(runs)}):\n")
        r_headers = ["KERNEL ID", "ACCEL", "SUBMITTED AT", "DURATION", "STATUS"]
        r_rows = []
        for r in runs[:10]:
            dur_str = f"{r['duration_sec']:.1f}s" if r["duration_sec"] else "-"
            r_rows.append(
                [
                    r["kernel_id"],
                    r["accelerator"],
                    r["submitted_at"],
                    dur_str,
                    r["status"].upper(),
                ]
            )
        r_widths = [len(h) for h in r_headers]
        for row in r_rows:
            for i, val in enumerate(row):
                r_widths[i] = max(r_widths[i], len(val))
        r_fmt = "  ".join(f"{{:<{w}}}" for w in r_widths)
        sys.stdout.write(r_fmt.format(*r_headers) + "\n")
        sys.stdout.write("  ".join("-" * w for w in r_widths) + "\n")
        for row in r_rows:
            sys.stdout.write(r_fmt.format(*row) + "\n")
        sys.stdout.write("\n")

    return 0


def handle_dataset_push(args: argparse.Namespace) -> int:
    """Handle 'dataset push' subcommand."""
    data_dir = Path(args.data_dir).resolve()
    if not data_dir.exists():
        logger.error("Dataset directory does not exist: %s", data_dir)
        return 1

    set_active_account(resolve_account(args.account))

    config = DatasetConfig(
        title=args.title or data_dir.name,
        slug=args.slug or data_dir.name,
        data_dir=data_dir,
        is_public=args.public,
        license_name=args.license,
    )

    if args.dry_run:
        user = get_authenticated_username()
        meta_file = create_dataset_metadata(config, user, data_dir)
        logger.info("Dry run: created metadata at %s", meta_file)
        return 0

    url = push_dataset(config, version_notes=args.notes)
    logger.info("Dataset available at: %s", url)
    return 0


def handle_exec(args: argparse.Namespace) -> int:
    """Handle 'exec' subcommand for interactive Jupyter proxy execution."""
    from kagglex.config import resolve_jupyter_url
    from kagglex.interactive import JupyterProxyClient

    url = resolve_jupyter_url(args.url, start_dir=find_repo_root())
    if not url:
        logger.error(
            "Kaggle Jupyter URL not provided. Please supply --url <URL> or set the "
            "KAGGLE_JUPYTER_URL environment variable.\n"
            "(In a running Kaggle notebook, click: Run -> Kaggle Jupyter Server -> Copy URL)"
        )
        return 1

    client = JupyterProxyClient(
        base_url=url,
        timeout=args.timeout,
        on_output=lambda chunk: sys.stdout.write(chunk),
    )

    if args.test:
        connected = client.test_connection()
        if connected:
            logger.info("Successfully connected to Kaggle Jupyter proxy server.")
            return 0
        else:
            logger.error(
                "Failed to connect to Kaggle Jupyter proxy server. Verify your URL and token."
            )
            return 1

    if args.gpu_info:
        info = client.get_gpu_info()
        sys.stdout.write(info + "\n")
        return 0

    if args.list_files is not None:
        target_dir = (
            args.list_files
            if isinstance(args.list_files, str) and args.list_files
            else ""
        )
        files = client.list_files(target_dir)
        for f in files:
            t = "[DIR] " if f.get("type") == "directory" else "      "
            sz = f"{f['size']}B" if f.get("size") is not None else ""
            sys.stdout.write(f"{t} {f.get('name', ''):<30} {sz}\n")
        return 0

    if args.upload:
        local_path = Path(args.upload).resolve()
        remote_name = args.remote_name or local_path.name
        ok = client.upload_file(local_path, remote_name)
        if ok:
            logger.info(
                "Uploaded %s to /kaggle/working/%s", local_path.name, remote_name
            )
            return 0
        else:
            logger.error("Failed to upload file %s", local_path)
            return 1

    if args.download:
        remote_name = args.download
        local_dest = Path(args.output or (Path.cwd() / Path(remote_name).name))
        client.download_file(remote_name, local_dest)
        logger.info("Downloaded %s to %s", remote_name, local_dest)
        return 0

    # Execute code
    code_to_run = args.code
    if not code_to_run and args.file:
        file_path = Path(args.file).resolve()
        if not file_path.is_file():
            logger.error("Specified script file does not exist: %s", file_path)
            return 1
        code_to_run = file_path.read_text(encoding="utf-8")

    if not code_to_run:
        logger.error(
            "No code or script specified to execute. Use: kagglex exec 'print(1)' or --file script.py"
        )
        return 1

    res = client.execute(code_to_run, timeout=args.timeout)
    sys.stdout.flush()
    return 0 if res.get("success") else 1


def _package_version() -> str:
    from importlib.metadata import PackageNotFoundError, version

    try:
        return version("kagglex")
    except PackageNotFoundError:
        return "unknown"


def create_parser() -> argparse.ArgumentParser:
    """Construct CLI argument parser for kagglex."""
    parser = argparse.ArgumentParser(
        prog="kagglex",
        description="Execute local Python code, modules, and experiments seamlessly on Kaggle GPUs.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Enable verbose debug logging"
    )
    parser.add_argument(
        "-V", "--version", action="version", version=f"%(prog)s {_package_version()}"
    )

    subparsers = parser.add_subparsers(dest="subcommand", required=True)

    # 1. 'run' subcommand
    run_p = subparsers.add_parser(
        "run",
        help="Stage, dispatch, monitor, and sync outputs from a Kaggle job.",
    )
    run_p.add_argument(
        "command_pos",
        nargs="?",
        default=None,
        metavar="COMMAND",
        help="Command to run (e.g. 'python train.py --epochs 10')",
    )
    run_p.add_argument(
        "--command",
        type=str,
        default=None,
        help="Explicit command string to run remotely",
    )
    run_p.add_argument(
        "--file",
        type=str,
        default=None,
        help="Target standalone Python script to run remotely",
    )
    run_p.add_argument(
        "--title",
        type=str,
        default=None,
        help="Kernel title",
    )
    run_p.add_argument(
        "--slug",
        type=str,
        default=None,
        help="Custom kernel slug",
    )
    run_p.add_argument(
        "--dir",
        type=str,
        default=None,
        help="Project directory to package (defaults to current working directory)",
    )
    run_p.add_argument(
        "--gpu",
        type=str,
        choices=["t4-2x", "p100", "v3-8", "none"],
        default="t4-2x",
        help="Accelerator type (default: 't4-2x')",
    )
    run_p.add_argument(
        "--multi-gpu",
        action="store_true",
        help="Automatically wrap command in torchrun across multiple GPUs",
    )
    run_p.add_argument(
        "--datasets",
        nargs="*",
        default=[],
        help="Kaggle dataset slugs to mount (e.g. 'user/dataset-name')",
    )
    run_p.add_argument(
        "--include-data",
        nargs="*",
        default=[],
        help="Local data files or folders to bundle into the payload",
    )
    run_p.add_argument(
        "--extra-deps",
        nargs="*",
        default=[],
        help="Additional pip packages to install remotely",
    )
    run_p.add_argument(
        "--env",
        nargs="*",
        default=[],
        help="Environment variables to inject (e.g. KEY=VAL or KEY)",
    )
    run_p.add_argument(
        "--env-file",
        type=str,
        default=None,
        help="Path to .env file to inject",
    )
    run_p.add_argument(
        "--kaggle-secrets",
        nargs="*",
        default=[],
        help="Kaggle secrets to inject (e.g. WANDB_API_KEY HF_TOKEN)",
    )
    run_p.add_argument(
        "--parent-kernels",
        nargs="*",
        default=[],
        help="Parent kernel slugs to attach outputs from",
    )
    run_p.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Local directory to download artifacts to upon completion",
    )
    run_p.add_argument(
        "--record-file",
        type=str,
        default=None,
        help="Markdown file (e.g. README.md) to append execution details",
    )
    run_p.add_argument(
        "--poll-interval",
        type=int,
        default=20,
        help="Seconds between status checks (default: 20s)",
    )
    run_p.add_argument(
        "--stream",
        action="store_true",
        help="Stream remote execution logs in real-time",
    )
    run_p.add_argument(
        "--no-wait",
        action="store_true",
        help="Submit job and exit immediately",
    )
    run_p.add_argument(
        "--no-internet",
        action="store_true",
        help="Disable internet inside kernel",
    )
    run_p.add_argument(
        "--include-outputs",
        nargs="*",
        default=[],
        help="Glob patterns of files to download",
    )
    run_p.add_argument(
        "--exclude-outputs",
        nargs="*",
        default=[],
        help="Glob patterns of files to exclude from download",
    )
    run_p.add_argument(
        "--auto-dataset",
        action="store_true",
        help="Automatically publish project payload as a private Kaggle dataset if size exceeds limit",
    )
    run_p.add_argument(
        "--auto-dataset-slug",
        type=str,
        default=None,
        help="Custom dataset slug for auto-dataset payload",
    )
    run_p.add_argument(
        "--dry-run",
        action="store_true",
        help="Stage all files locally without submitting to Kaggle",
    )
    run_p.add_argument(
        "--account",
        type=str,
        default=None,
        metavar="NAME",
        help="Named Kaggle account to run on, or 'auto' for the one with most "
        "remaining quota (default: the default account)",
    )

    # 2. 'push' subcommand
    push_p = subparsers.add_parser(
        "push",
        help="Stage and submit a job to Kaggle without waiting for results.",
    )
    for action in run_p._actions:
        if action.dest not in {"help", "subcommand", "no_wait", "stream"}:
            push_p._add_action(action)

    kernel_account_help = (
        "Kaggle account to use (default: the account that submitted the run)"
    )

    # 3. 'status' subcommand
    status_p = subparsers.add_parser("status", help="Check status of a Kaggle kernel.")
    status_p.add_argument("kernel", type=str, help="Kernel ID or slug")
    status_p.add_argument("--account", metavar="NAME", help=kernel_account_help)

    # 4. 'logs' subcommand
    logs_p = subparsers.add_parser("logs", help="View remote logs for a Kaggle kernel.")
    logs_p.add_argument("kernel", type=str, help="Kernel ID or slug")
    logs_p.add_argument("--account", metavar="NAME", help=kernel_account_help)

    # 5. 'cancel' subcommand
    cancel_p = subparsers.add_parser("cancel", help="Cancel a running Kaggle kernel.")
    cancel_p.add_argument("kernel", type=str, help="Kernel ID or slug")
    cancel_p.add_argument("--account", metavar="NAME", help=kernel_account_help)

    # 6. 'list' subcommand
    list_p = subparsers.add_parser(
        "list", help="List recent Kaggle runs (or accounts with --credit)."
    )
    list_p.add_argument(
        "--limit", type=int, default=20, help="Number of runs to display (default: 20)"
    )
    list_p.add_argument(
        "--credit",
        action="store_true",
        help="Show Kaggle accounts and their estimated GPU/TPU quota instead "
        "(same as 'kagglex account list')",
    )

    # 7. 'pull' subcommand
    pull_p = subparsers.add_parser(
        "pull", help="Download outputs from completed kernel."
    )
    pull_p.add_argument("kernel", type=str, help="Kernel ID or slug")
    pull_p.add_argument("--account", metavar="NAME", help=kernel_account_help)
    pull_p.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory to save downloaded files",
    )
    pull_p.add_argument(
        "--include-outputs",
        nargs="*",
        default=[],
        help="Glob patterns of files to download",
    )
    pull_p.add_argument(
        "--exclude-outputs",
        nargs="*",
        default=[],
        help="Glob patterns of files to exclude",
    )

    # 8. 'dataset' subcommand
    ds_p = subparsers.add_parser("dataset", help="Manage Kaggle datasets.")
    ds_sub = ds_p.add_subparsers(dest="dataset_action", required=True)

    ds_push = ds_sub.add_parser(
        "push", help="Create or update a Kaggle dataset from local directory."
    )
    ds_push.add_argument(
        "--data-dir", type=str, required=True, help="Local directory containing data"
    )
    ds_push.add_argument(
        "--title",
        type=str,
        default=None,
        help="Dataset title (defaults to folder name)",
    )
    ds_push.add_argument(
        "--slug", type=str, default=None, help="Dataset slug (defaults to folder name)"
    )
    ds_push.add_argument(
        "--public", action="store_true", help="Make dataset public (default: private)"
    )
    ds_push.add_argument(
        "--license", type=str, default="CC0-1.0", help="Dataset license"
    )
    ds_push.add_argument(
        "--notes", type=str, default="Upload dataset", help="Version description notes"
    )
    ds_push.add_argument(
        "--dry-run",
        action="store_true",
        help="Stage dataset metadata locally without pushing",
    )
    ds_push.add_argument(
        "--account",
        metavar="NAME",
        help="Kaggle account that will own the dataset (default: the default account)",
    )

    # 9. 'exec' subcommand for interactive proxy execution
    exec_p = subparsers.add_parser(
        "exec",
        help="Execute code interactively on an active Kaggle Jupyter session via proxy URL.",
    )
    exec_p.add_argument(
        "code",
        nargs="?",
        default=None,
        help="Python code string to execute (e.g. 'import torch; print(torch.cuda.is_available())')",
    )
    exec_p.add_argument(
        "--url",
        type=str,
        default=None,
        help="Kaggle Jupyter proxy URL (default: KAGGLE_JUPYTER_URL env var)",
    )
    exec_p.add_argument(
        "--file",
        type=str,
        default=None,
        help="Path to local Python script to execute remotely",
    )
    exec_p.add_argument(
        "--timeout",
        type=int,
        default=120,
        help="Execution timeout in seconds (default: 120s)",
    )
    exec_p.add_argument(
        "--test",
        action="store_true",
        help="Test connection to the Kaggle Jupyter proxy server",
    )
    exec_p.add_argument(
        "--gpu-info",
        action="store_true",
        help="Query GPU status and device names on the active kernel",
    )
    exec_p.add_argument(
        "--list-files",
        nargs="?",
        const="",
        default=None,
        help="List files in remote /kaggle/working/ directory",
    )
    exec_p.add_argument(
        "--upload",
        type=str,
        default=None,
        help="Upload local file to /kaggle/working/",
    )
    exec_p.add_argument(
        "--remote-name",
        type=str,
        default=None,
        help="Destination filename when uploading",
    )
    exec_p.add_argument(
        "--download",
        type=str,
        default=None,
        help="Download file from /kaggle/working/",
    )
    exec_p.add_argument(
        "-o",
        "--output",
        type=str,
        default=None,
        help="Local destination path for downloaded file",
    )

    # 10. 'quota' subcommand
    quota_p = subparsers.add_parser(
        "quota",
        help="Inspect local GPU/TPU quota usage and estimated remaining hours.",
    )
    quota_p.add_argument(
        "--days",
        type=int,
        default=7,
        help="Rolling time window in days (default: 7)",
    )
    quota_p.add_argument(
        "--gpu-limit",
        type=float,
        default=30.0,
        help="Weekly GPU quota limit in hours (default: 30.0)",
    )
    quota_p.add_argument(
        "--tpu-limit",
        type=float,
        default=20.0,
        help="Weekly TPU quota limit in hours (default: 20.0)",
    )
    quota_p.add_argument(
        "--account",
        metavar="NAME",
        help="Only count runs of this named account (default: all runs)",
    )

    # 11. 'account' subcommand
    acct_p = subparsers.add_parser(
        "account",
        help="Manage named Kaggle accounts (add, test, list, use, remove).",
    )
    acct_sub = acct_p.add_subparsers(dest="account_action", required=True)

    acct_add = acct_sub.add_parser(
        "add",
        help="Register a Kaggle account under a name (prompts for the key if omitted).",
    )
    acct_add.add_argument("name", help="Friendly account name, e.g. Phong")
    acct_add.add_argument(
        "--username", help="Kaggle username (required unless --json is used)"
    )
    acct_secret = acct_add.add_mutually_exclusive_group()
    acct_secret.add_argument(
        "--key", help="Legacy API key from kaggle.json (prefer the hidden prompt)"
    )
    acct_secret.add_argument(
        "--token", help="New-style Kaggle API token (prefer the hidden prompt)"
    )
    acct_secret.add_argument(
        "--json",
        metavar="PATH",
        help="Import username and key from a kaggle.json file",
    )
    acct_add.add_argument(
        "--default", action="store_true", help="Make this the default account"
    )
    acct_add.add_argument(
        "--force",
        action="store_true",
        help="Replace an existing account with this name",
    )
    acct_add.add_argument(
        "--verify",
        action="store_true",
        help="Test the connection to Kaggle right after saving",
    )

    acct_test = acct_sub.add_parser(
        "test", help="Test the connection to Kaggle for accounts (default: all)."
    )
    acct_test.add_argument(
        "names", nargs="*", metavar="NAME", help="Account names to test"
    )

    acct_list = acct_sub.add_parser(
        "list", help="Show accounts with estimated GPU/TPU quota used and remaining."
    )
    acct_list.add_argument(
        "--days", type=int, default=None, help="Rolling window in days (default: 7)"
    )
    acct_list.add_argument(
        "--gpu-limit",
        type=float,
        default=None,
        help="Weekly GPU quota per account in hours (default: 30)",
    )
    acct_list.add_argument(
        "--tpu-limit",
        type=float,
        default=None,
        help="Weekly TPU quota per account in hours (default: 20)",
    )

    acct_use = acct_sub.add_parser("use", help="Set the default account.")
    acct_use.add_argument("name", help="Account name")

    acct_rm = acct_sub.add_parser("remove", help="Delete a stored account.")
    acct_rm.add_argument("name", help="Account name")

    parser.epilog = _build_flag_reference(subparsers)
    return parser


def _build_flag_reference(subparsers: argparse._SubParsersAction) -> str:
    """Render every subcommand with its arguments for the top-level --help."""
    lines = ["full reference (run 'kagglex <command> --help' for details):", ""]

    def describe(prefix: str, sub: argparse._SubParsersAction) -> None:
        helps = {a.dest: a.help for a in sub._choices_actions}
        for name, p in sub.choices.items():
            lines.append(f"  kagglex {prefix}{name}  -  {helps.get(name, '')}")
            nested = None
            for action in p._actions:
                if isinstance(action, argparse._SubParsersAction):
                    nested = action
                    continue
                if isinstance(action, argparse._HelpAction):
                    continue
                if action.option_strings:
                    label = ", ".join(action.option_strings)
                    if action.nargs != 0:
                        label += f" {action.metavar or action.dest.upper()}"
                else:
                    label = action.metavar or action.dest.upper()
                lines.append(f"      {label:<28} {action.help or ''}")
            if nested is not None:
                describe(f"{prefix}{name} ", nested)
            lines.append("")

    describe("", subparsers)
    lines += [
        "examples:",
        "  kagglex account add Phong --username phong_kg      # prompts for key",
        "  kagglex list --credit",
        "  kagglex run --file train.py --account auto --gpu t4-2x",
    ]
    return "\n".join(lines)


def main(args_list: list[str] | None = None) -> int:
    """Main CLI entrypoint."""
    parser = create_parser()
    args = parser.parse_args(args_list)
    setup_logging(args.verbose)

    # Handle command positional vs flag
    if hasattr(args, "command_pos") and args.command_pos and not args.command:
        args.command = args.command_pos

    try:
        if args.subcommand in {"run", "push"}:
            return handle_run(args)
        elif args.subcommand == "exec":
            return handle_exec(args)
        elif args.subcommand == "status":
            return handle_status(args)
        elif args.subcommand == "logs":
            return handle_logs(args)
        elif args.subcommand == "cancel":
            return handle_cancel(args)
        elif args.subcommand == "list":
            return handle_list(args)
        elif args.subcommand == "quota":
            return handle_quota(args)
        elif args.subcommand == "pull":
            return handle_pull(args)
        elif args.subcommand == "dataset":
            if args.dataset_action == "push":
                return handle_dataset_push(args)
        elif args.subcommand == "account":
            return {
                "add": handle_account_add,
                "list": handle_account_list,
                "use": handle_account_use,
                "remove": handle_account_remove,
                "test": handle_account_test,
            }[args.account_action](args)
        else:
            parser.print_help()
            return 1
    except AccountError as e:
        logger.error("%s", e)
        return 1
    except Exception as e:
        logger.exception("Operation failed: %s", e)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
