# kagglex

Execute local Python code, modules, and experiments seamlessly on Kaggle GPUs and TPUs.

[![PyPI version](https://img.shields.io/pypi/v/kagglex)](https://pypi.org/project/kagglex/)
[![Downloads](https://pepy.tech/badge/kagglex)](https://pepy.tech/projects/kagglex)
[![Python](https://img.shields.io/pypi/pyversions/kagglex)](https://pypi.org/project/kagglex/)
[![License](https://img.shields.io/pypi/l/kagglex)](https://github.com/Hari31416/kagglex/blob/main/LICENSE)
[![CI](https://github.com/Hari31416/kagglex/actions/workflows/ci.yml/badge.svg)](https://github.com/Hari31416/kagglex/actions/workflows/ci.yml)

> **This is a fork** of [Hari31416/kagglex](https://github.com/Hari31416/kagglex) (MIT). It adds multi-account support for teams that pool Kaggle accounts: named accounts, per-account GPU/TPU quota estimates, and `--account auto`. See [Multiple Kaggle Accounts](docs/user-guide/accounts.md).
>
> Install this fork (needs [uv](https://docs.astral.sh/uv/)):
>
> ```bash
> uv tool install git+https://github.com/taita2005/kagglex-tai-custom.git
> kagglex account add <name> --username <kaggle-username> --verify   # prompts for the API token
> kagglex list --credit
> ```
>
> Each person's accounts and tokens stay on their own machine in `~/.kagglex/accounts.toml`. Nothing is shared through this repo.

## Overview

`kagglex` enables machine learning practitioners and researchers to transparently package and dispatch local Python code, standalone scripts, or complete packages to Kaggle's cloud GPU and TPU environments without tedious manual uploading or notebook maintenance.

## Features

- Flexible project detection for standalone scripts, flat packages, or `src/` layout projects
- Pre-flight validation with local AST syntax checking and credential verification
- Automatic ignore filtering powered by `pathspec` supporting `.gitignore` and `.kaggleignore`
- Dual-tier packaging preventing bloated uploads and base64 truncation
- Automated multi-GPU execution using `torchrun`
- Secure Kaggle Secrets integration for WandB, HuggingFace, and custom credentials
- Local experiment history repository for tracking past runs, statuses, and durations
- Interactive REPL execution on active Kaggle notebooks via Jupyter proxy URL
- Multiple named Kaggle accounts with per-account quota estimates and automatic account selection
- Programmatic Python SDK alongside the `kagglex` CLI

## Installation

Install via uv or pip:

```bash
uv pip install kagglex
```

## CLI Usage

### Run a Standalone Script

```bash
kagglex run --file train.py --gpu t4-2x --title "Pilot Training"
```

### Run a Module with Multi-GPU

```bash
kagglex run "python -m mypkg.train --epochs 10" --gpu t4-2x --multi-gpu
```

### Stream Remote Logs in Real-Time

```bash
kagglex run "python train.py" --stream
```

### List Recent Runs

```bash
kagglex list
```

### Check Status or Cancel a Run

```bash
kagglex status my-experiment
kagglex cancel my-experiment
```

Note: Kaggle's public API does not support programmatic session cancellation for running kernels. `kagglex cancel` marks the local run record as cancelled and provides the direct Kaggle URL to stop the session in the web interface.

### Pull Downloaded Outputs

```bash
kagglex pull my-experiment --output-dir ./results --include-outputs "*.json" "checkpoints/*"
```

### Push a Kaggle Dataset

```bash
kagglex dataset push --data-dir ./data/embeddings --title "Embeddings Dataset"
```

### Auto-Dataset Payload Offloading

When your project files or local assets exceed the inline payload limit (5 MB), automatically stage and upload them as a private Kaggle dataset:

```bash
kagglex run --file train.py --auto-dataset
```

### Inspect GPU and TPU Quota Usage

Track your rolling 7-day accelerator consumption against Kaggle's weekly quotas (30 GPU hours / 20 TPU hours):

```bash
kagglex quota --days 7
```

Because Kaggle's public API does not expose an endpoint to query remaining weekly quota balances, `kagglex` computes usage locally from run records in `~/.kagglex/runs.json` over a configurable rolling window (default: 7 days). Limit thresholds can be configured via `--gpu-limit` / `--tpu-limit` or in configuration files.

### Multiple Kaggle Accounts

Store several Kaggle accounts under friendly names (handy for teams that pool GPU hours) and pick one per run:

```bash
kagglex account add Phong --username phong_kg      # prompts for the API key/token
kagglex account add Tân --json ~/Downloads/kaggle.json
kagglex list --credit                              # accounts + estimated GPU/TPU left
kagglex run --file train.py --account Tân          # or --account auto
kagglex account use Tân                            # change the default account
```

Credentials live in `~/.kagglex/accounts.toml` (chmod 600). `status`, `logs`, `pull` and `cancel` automatically use the account that submitted the run.

### Declarative Configuration

`kagglex` supports a multi-tier configuration hierarchy (later sources override earlier ones):

- **User Global**: `~/.kagglex/config.toml` (or `~/.kagglex/kagglex.toml`)
- **Project Settings**: `pyproject.toml` (`[tool.kagglex]`) or `kagglex.toml`
- **CLI Arguments**: Command line flags override all configuration files

Example project configuration in `pyproject.toml`:

```toml
[tool.kagglex]
gpu = "t4-2x"
multi_gpu = true
kaggle_secrets = ["WANDB_API_KEY", "HF_TOKEN"]
datasets = ["username/my-dataset"]
include_outputs = ["*.json", "checkpoints/*"]

[tool.kagglex.env]
WANDB_PROJECT = "my-experiment"
```

Or global defaults in `~/.kagglex/config.toml`:

```toml
gpu = "p100"
quota_days = 7
gpu_weekly_limit_hours = 30.0

[env]
WANDB_ENTITY = "my-team"
```

Then simply run:

```bash
kagglex run --file train.py
```

### Interactive REPL on Running Kaggle Notebooks

When a notebook is already open in Kaggle, copy its proxy URL (`Run -> Kaggle Jupyter Server -> Copy URL`) and run commands with sub-second feedback:

```bash
# Verify connection
kagglex exec --url "https://kkb-production.jupyter-proxy.kaggle.net?token=..." --test

# Query remote GPU status
kagglex exec --url "https://kkb-production.jupyter-proxy.kaggle.net?token=..." --gpu-info

# Execute inline Python snippets
kagglex exec "import torch; print(torch.cuda.device_count())"

# Execute a local Python file remotely
kagglex exec --file evaluate.py

# List and transfer files
kagglex exec --list-files
kagglex exec --upload ./checkpoint.pt
kagglex exec --download run_results.json -o ./local_results.json
```

Alternatively, set the environment variable:

```bash
export KAGGLE_JUPYTER_URL="https://kkb-production.jupyter-proxy.kaggle.net?token=..."
```

## Python SDK Usage

```python
from kagglex import KaggleRunner, RunConfig

runner = KaggleRunner()
job = runner.run(
    command="python -m mypkg.train --batch-size 64",
    title="Fine Tuning Run",
    gpu="t4-2x",
    multi_gpu=True,
    wait=True,
)

print(f"Status: {job.status}")
job.pull_outputs(destination_dir="./results")
```
