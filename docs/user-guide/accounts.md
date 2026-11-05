# Multiple Kaggle Accounts

Teams (for example in a hackathon) often pool several Kaggle accounts to get more GPU/TPU hours. `kagglex` can store any number of accounts under friendly names and pick one per run.

## Registering accounts

```bash
# Prompts for the API key/token with hidden input (keeps it out of shell history)
kagglex account add Phong --username phong_kg

# Or import a teammate's kaggle.json
kagglex account add Tân --json ~/Downloads/kaggle.json

# Or pass the secret explicitly
kagglex account add Minh --username minh_kg --key 0123456789abcdef0123456789abcdef
kagglex account add Lan  --username lan_kg  --token KGAT_xxxxxxxx
```

- The name (`Phong`) is only a label. `--username` must be the real Kaggle username.
- Both credential types are supported. Legacy API keys (`key` in `kaggle.json`, 32 hex characters) use `--key`. New-style API tokens use `--token`. When you type the secret at the prompt, its type is detected automatically.
- Add `--verify` to test the connection right after saving. For tokens this also checks that the token belongs to `--username`.
- The first account you add becomes the default. Use `--default` to make a later one the default, and `--force` to replace an existing account.
- Name lookups ignore case and accept Vietnamese diacritics, so `tân` matches `Tân`.

Accounts are stored in `~/.kagglex/accounts.toml` with owner-only permissions (`chmod 600`):

```toml
default = "Phong"

[accounts."Phong"]
username = "phong_kg"
key = "0123456789abcdef0123456789abcdef"

[accounts."Tân"]
username = "tan_kg"
token = "KGAT_xxxxxxxx"
```

## Viewing accounts and remaining quota

```bash
kagglex account list        # or: kagglex list --credit
```

```text
Kaggle accounts - estimated quota, past 7 days:

   NAME   KAGGLE USER  AUTH   GPU USED  GPU LEFT   TPU USED  TPU LEFT   RUNS
-  -----  -----------  -----  --------  ---------  --------  ---------  ----
*  Phong  phong_kg     key    12.50h    17.50/30h  0.00h     20.00/20h  4
   Tân    tan_kg       token  3.00h     27.00/30h  2.00h     18.00/20h  2
```

`*` marks the default account. Options: `--days`, `--gpu-limit` and `--tpu-limit`. The limits can also be set with `gpu_weekly_limit_hours` / `tpu_weekly_limit_hours` in your configuration files.

!!! warning "Usage is an estimate"
    Kaggle has no API that reports remaining quota. Usage is summed from the runs in `~/.kagglex/runs.json`, where each run is counted against the account that owns the kernel (the `username/` prefix of the kernel id). A run's duration is only recorded when kagglex waited for it to finish. Runs started with `--no-wait`/`push`, notebooks run on the Kaggle website, and runs submitted from other machines are not counted.

## Choosing the account for a run

```bash
kagglex run --file train.py --account Tân     # a specific account
kagglex run --file train.py --account auto    # account with most remaining quota
kagglex run --file train.py                   # the default account
```

The account is chosen in this order: `--account` on the command line, then `account = "..."` under `[tool.kagglex]` (or in `kagglex.toml`), then the default account, then Kaggle's own credentials (`~/.kaggle/kaggle.json` or environment variables) if no accounts are registered.

`auto` compares remaining GPU hours, or TPU hours for `--gpu v3-8`. If two accounts tie, the default account wins.

The account is saved with the run. `kagglex status`, `logs`, `pull` and `cancel` then automatically use the account that owns the kernel, and all of them accept `--account NAME` to override it. `kagglex quota --account NAME` shows the detailed quota view for a single account. `kagglex dataset push --account NAME` uploads a dataset under that account.

## Managing accounts

```bash
kagglex account test          # check every account can reach Kaggle (or: test Phong Tân)
kagglex account use Tân       # change the default
kagglex account remove Phong  # delete stored credentials
```

## Python API

```python
from kagglex import KaggleRunner, RunConfig

runner = KaggleRunner()
job = runner.run(command="python train.py", title="Exp", account="auto")
```
