# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

`kagglex` is a CLI (`kagglex`) and Python SDK (`KaggleRunner`) that packages local Python code and runs it as a private Kaggle script kernel on GPU/TPU, then polls, streams logs and pulls outputs back. Python >=3.10, managed with `uv`, built with hatchling.

## Commands

```bash
make test                                   # uv run pytest (what CI runs)
uv run pytest tests/test_packager.py        # single file
uv run pytest tests/test_cli.py::test_name  # single test
uv run pytest -k "auto_dataset"             # by keyword
make test-cov                               # coverage for kagglex
make check                                  # black --check + ruff (no edits)
make fix                                    # black + ruff --fix
make docs-serve                             # mkdocs (uses the `docs` dependency group)
make build                                  # uv build
```

Pre-commit runs black, ruff --fix and a 5 MB large-file check. Tests are fully offline: the `mock_kaggle_api` fixture in `tests/conftest.py` patches `get_kaggle_api` in both `kagglex.client` and `kagglex.dataset`. If you add a module that imports `get_kaggle_api`, patch it there too. An autouse fixture (`isolated_home`) points `HOME` at a temp dir and resets the active account, so tests never touch the real `~/.kagglex` or `~/.kaggle`.

## Architecture

The main `run` flow is in `KaggleRunner.stage()` / `run()` in `src/kagglex/api.py`. The CLI (`cli.py`) mostly builds a `RunConfig` and calls into it.

1. **Config resolution** (`config.py::load_project_config`): global `~/.kagglex/config.toml` (or `kagglex.toml`, or `~/.config/kagglex/config.toml`), then `[tool.kagglex]` in `pyproject.toml`, then `kagglex.toml`. Each later source overrides the earlier ones, and nested dicts such as `env` are deep-merged. The CLI applies its flags last in `cli.py::handle_run`. A flag counts as unset when it equals its argparse default (for example `--gpu t4-2x` or `--poll-interval 20`), so config-file values can win over an explicitly passed default. Keep this in mind when you add flags.
2. **Workspace** (`workspace.py`): `find_repo_root` walks up looking for project markers. `detect_project_type` chooses between standalone script, `src/` layout, flat package and plain directory. A pre-flight AST syntax check runs on the target script.
3. **Packaging** (`packager.py`, `ignore.py`): the project is zipped into `pkg_payload.zip`. Optional `--include-data` files go into `data_payload.zip`. Files are filtered through `.gitignore` and `.kaggleignore` using `pathspec`. Each zip is capped at `MAX_PAYLOAD_BYTES` (5 MB) unless `auto_dataset` is set. `create_kernel_metadata` writes `kernel-metadata.json`, prefixing the username onto dataset and kernel refs.
4. **Bootstrap** (`bootstrap.py`): `BOOTSTRAP_TEMPLATE` is a complete Python script that runs **on Kaggle**, not locally. Parameters are injected by replacing `__NAME_JSON__` placeholders, and the zips are inlined as base64. On Kaggle the script:
   - loads Kaggle Secrets
   - extracts the payload from inline base64, falling back to `/kaggle/input/*/pkg_payload.zip`
   - runs `pip install -e`, or installs `requirements.txt`
   - wraps the command in `torchrun` when `multi_gpu` is set and more than one GPU is present
   - tees output to `/kaggle/working/outputs/execution.log` and `run_summary.json`

   The template has to stay self-contained, using only the stdlib plus optional torch.
5. **Auto-dataset offload**: when `auto_dataset` is set, payloads over 5 MB are left out of the inline base64. They are copied into a private dataset (`kagglex-payload-<slug>`), which is pushed via `dataset.py` before the kernel push and added to `dataset_slugs`.
6. **Kaggle API** (`client.py`): wraps the `kaggle` package for push, status polling, log streaming and output pull. When pulling outputs, shell globs are converted to regexes for server-side filtering. `DEFAULT_OUTPUT_EXCLUDES` skips `*.pt`, payload zips, `pkg_src` and similar files unless they are explicitly included.
7. **History** (`history.py`): every run is recorded only in `~/.kagglex/runs.json` (legacy `~/.kagglerun` is migrated), with `project_path` for provenance. `kagglex quota` calculates GPU/TPU hours from this file over a rolling window, because Kaggle has no quota API. `cancel` only marks the local record, since Kaggle's public API can't stop a running kernel.

8. **Accounts** (`accounts.py`): named accounts live in `~/.kagglex/accounts.toml` (written atomically, chmod 600). Each account has a Kaggle username plus either a legacy `key` or a new-style `token`.
   - Commands resolve an account and call `set_active_account()`. `client.get_kaggle_api()` reads this process-wide active account; with none set, it falls back to Kaggle's own credential lookup.
   - `_apply_account_credentials` sets `config_values` directly instead of calling `authenticate()`. For key-based accounts it also overrides kagglesdk's `_try_fill_auth`, because kaggle 2.x prefers `KAGGLE_API_TOKEN` / `~/.kaggle/access_token` over username+key and would otherwise silently use the wrong account. `test_legacy_key_wins_over_stray_access_token` guards this.
   - Quota is attributed per account through the kernel-id owner prefix (`get_quota_usage(username=...)`). `record_run` therefore always prepends a new record and never replaces one.
   - `--account auto` picks the account with the most remaining quota for the accelerator.

Staging artifacts are written to `<repo>/.kagglex/staging/<slug>/`.

`interactive.py` (`kagglex exec`) is a separate path that does not use the kernel-push pipeline. It talks to an already-running Kaggle notebook through its Jupyter proxy URL, using REST plus a WebSocket. The URL comes from `--url`, `KAGGLE_JUPYTER_URL` or the `url`/`jupyter_url` config keys.

Accelerators are validated against `SUPPORTED_ACCELERATORS` in `config.py`: `t4-2x`, `p100`, `v3-8` (TPU) and `none`.

## Conventions

- Commits use Conventional Commits with a scope, e.g. `feat(backend): ...`, `docs(general): ...`, `chore(general): release version X.Y.Z`.
- Releases: bump `version` in `pyproject.toml` and add a `## [X.Y.Z] - date` section to `CHANGELOG.md` (Keep a Changelog format). Pushing a `vX.Y.Z` tag runs `release.yml`, which tests, builds, creates a GitHub release from that changelog section and publishes to PyPI.
- When behavior changes, update user-facing docs in both `README.md` and `docs/`, which is deployed by MkDocs.
- A user-level skill at `~/.claude/skills/kagglex/` teaches other sessions how to use the CLI. Whenever you change commands or flags, regenerate its `reference.md` from `kagglex --help`, and update `SKILL.md` if the workflow or gotchas change.
