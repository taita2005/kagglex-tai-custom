# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Named multi-account support: `kagglex account add | list | use | remove`, with credentials stored in `~/.kagglex/accounts.toml` (chmod 600). Both legacy API keys and new-style API tokens are supported.
- `--account NAME|auto` on `run`/`push` (`auto` picks the account with the most remaining estimated quota), `[tool.kagglex] account = "..."` config key, and `--account` on `status`, `logs`, `pull`, `cancel`, `quota` and `dataset push`.
- `kagglex account test` and `account add --verify` check credentials against Kaggle, including whether a token belongs to the stored username.
- `kagglex list --credit` shows every account with its estimated GPU/TPU usage and remaining hours.
- `status`/`logs`/`pull`/`cancel` automatically use the account that submitted the run. Run history now records the account.
- `kagglex --help` now includes a full reference of every command and flag. Added `kagglex --version`.

### Changed

- Minimum supported Python is now 3.10. The code already required it.
- Quota usage can be filtered per Kaggle username (`get_quota_usage(username=...)`).

### Fixed

- Declare `six` as a dependency: kaggle 2.2.5 imports it without declaring it, which broke `uv tool install kagglex`.
- Re-running a kernel with the same slug no longer overwrites the previous run record, which used to erase its duration from quota tracking.
- A stray `KAGGLE_API_TOKEN` or `~/.kaggle/access_token` can no longer override the credentials of the selected account.

## [0.4.0] - 2026-09-19

### Added

- MkDocs documentation site with Material theme, search, and automated GitHub Pages deployment workflow.
- Complete documentation guides for quickstart, batch execution, interactive REPL, datasets, quota tracking, configuration, workspaces, CLI, and Python API.
- Support for user global configuration in `~/.kagglex/config.toml` (and `~/.kagglex/kagglex.toml`) with project-level overrides in `pyproject.toml` and `kagglex.toml`.
- Hierarchical merging of configuration dictionaries for environment variables across global, project, and CLI tiers.

### Changed

- Centralized experiment run history storage exclusively in `~/.kagglex/runs.json` with project provenance (`project_path`), eliminating project-local directory dual writes.

## [0.3.0] - 2026-09-19

### Added

- Automated large payload offloading (`--auto-dataset`, `--auto-dataset-slug`) to package and upload projects or local assets exceeding 5 MB as private Kaggle datasets.
- Remote bootstrap orchestrator support for detecting and extracting package and data payloads from mounted dataset inputs.
- Declarative project configuration support via `[tool.kagglex]` in `pyproject.toml`, project-level `kagglex.toml`, and user-level `~/.kagglex/config.toml`.
- GPU and TPU quota tracking command (`kagglex quota`) aggregating rolling 7-day accelerator consumption against Kaggle weekly limits.
- Pre-flight quota warnings before dispatching kernel runs when quota is nearing exhaustion (> 90% used or < 2 hours remaining).
- Safe TOML loader support using `tomllib` (Python 3.11+) and `tomli` fallback (Python < 3.11).

## [0.2.0] - 2026-09-17

### Fixed

- Convert shell glob patterns to regular expressions for Kaggle API server-side filtering, resolving silent regex syntax errors when pulling wildcard patterns (e.g. `*.json`).
- Prevent unneeded network downloads of large `.pt` checkpoints and `.pyc` files during default output pulls via server-side negative lookaheads.
- Raise an explicit `ValueError` when `pkg_payload.zip` or `data_payload.zip` exceeds 5 MB instead of silently dropping the payload.

### Changed

- Set payload size limit to 5 MB during project and local data packaging to prevent oversized inline kernel payloads.

## [0.1.0] - 2026-09-16

### Added

- Initial release of `kagglex` CLI and Python toolkit.
- Automatic packaging of local code and dependencies for Kaggle kernels.
- Remote execution on Kaggle GPUs and CPUs.
- Live output streaming and artifact synchronization.
- Pre-flight workspace validation and `.kaggleignore` filtering.
- Dataset upload and staging management.
