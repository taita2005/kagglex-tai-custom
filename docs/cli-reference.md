# CLI Reference

Reference of all `kagglex` commands, flags, and options.

## Global Options

| Option          | Description                   |
| --------------- | ----------------------------- |
| `-v, --verbose` | Enable debug logging output   |
| `-q, --quiet`   | Suppress non-error log output |
| `--help`        | Show help message and exit, plus a full reference of every command and flag |
| `-V, --version` | Print the installed kagglex version |

## kagglex run

Stage, submit, monitor, and retrieve outputs for a Kaggle experiment.

```bash
kagglex run [OPTIONS]
```

### Options

| Flag                        | Type                | Default        | Description                                           |
| --------------------------- | ------------------- | -------------- | ----------------------------------------------------- |
| `--file PATH`               | Path                | None           | Standalone Python script to execute                   |
| `--dir PATH`                | Path                | Current Dir    | Root directory of project to package                  |
| `--command CMD`             | String              | None           | Command to run remotely (e.g. `python -m pkg.train`)  |
| `--title TITLE`             | String              | Auto-generated | Title of the Kaggle notebook/kernel                   |
| `--slug SLUG`               | String              | Auto-generated | Slug for the Kaggle kernel (5-50 chars)               |
| `--gpu TYPE`                | String              | `t4-2x`        | Hardware accelerator: `t4-2x`, `p100`, `v3-8`, `none` |
| `--multi-gpu`               | Flag                | False          | Enable multi-GPU distributed training                 |
| `--dataset DATASET`         | String (repeatable) | `[]`           | Kaggle dataset reference (`username/slug`)            |
| `--include-data PATH`       | Path (repeatable)   | `[]`           | Local file or directory to package                    |
| `--extra-deps PKG`          | String (repeatable) | `[]`           | Extra pip dependencies to install remotely            |
| `--env KEY=VAL`             | String (repeatable) | `[]`           | Remote environment variable                           |
| `--env-file PATH`           | Path                | None           | Path to `.env` file                                   |
| `--secret NAME`             | String (repeatable) | `[]`           | Kaggle User Secret name                               |
| `--parent-kernel ID`        | String (repeatable) | `[]`           | Kernel output to mount as input                       |
| `--auto-dataset`            | Flag                | False          | Auto-publish large payload as a Kaggle dataset        |
| `--auto-dataset-slug SLUG`  | String              | None           | Custom slug for auto-dataset payload                  |
| `--no-internet`             | Flag                | False          | Disable internet access in Kaggle environment         |
| `--output-dir PATH`         | Path                | `./outputs`    | Directory to save downloaded run outputs              |
| `--include-outputs PATTERN` | String (repeatable) | `[]`           | Glob pattern for files to download                    |
| `--exclude-outputs PATTERN` | String (repeatable) | `[]`           | Glob pattern for files to ignore                      |
| `--no-wait`                 | Flag                | False          | Return immediately after submitting kernel            |
| `--stream`                  | Flag                | False          | Stream remote logs in real-time                       |
| `--no-pull`                 | Flag                | False          | Skip downloading outputs after completion             |
| `--dry-run`                 | Flag                | False          | Stage project locally without submitting to Kaggle    |
| `--account NAME`            | String              | Default account | Named account to run on, or `auto` (most remaining quota) |

## kagglex exec

Interact with a live running Kaggle Jupyter proxy session.

```bash
kagglex exec [OPTIONS] [CODE]
```

### Options

| Flag                | Type    | Default               | Description                                |
| ------------------- | ------- | --------------------- | ------------------------------------------ |
| `CODE`              | String  | None                  | Inline Python code to execute remotely     |
| `--url URL`         | String  | `$KAGGLE_JUPYTER_URL` | Full URL of Kaggle Jupyter Proxy           |
| `--file PATH`       | Path    | None                  | Local Python script to execute remotely    |
| `--test`            | Flag    | False                 | Test connection and kernel health          |
| `--gpu-info`        | Flag    | False                 | Display remote GPU device details and VRAM |
| `--upload PATH`     | Path    | None                  | Upload local file to `/kaggle/working`     |
| `--download NAME`   | String  | None                  | Download file from `/kaggle/working`       |
| `-o, --output PATH` | Path    | None                  | Destination path for downloaded file       |
| `--list-files`      | Flag    | False                 | List files in `/kaggle/working`            |
| `--timeout SEC`     | Integer | `120`                 | Request timeout in seconds                 |

## kagglex quota

Display estimated GPU and TPU accelerator consumption over a rolling window.

```bash
kagglex quota [OPTIONS]
```

### Options

| Flag                | Type    | Default | Description                                      |
| ------------------- | ------- | ------- | ------------------------------------------------ |
| `--days DAYS`       | Integer | `7`     | Number of days in the rolling calculation window |
| `--gpu-limit HOURS` | Float   | `30.0`  | Weekly GPU quota limit in hours                  |
| `--tpu-limit HOURS` | Float   | `20.0`  | Weekly TPU quota limit in hours                  |
| `--account NAME`    | String  | All     | Only count runs of this named account            |

## kagglex list

List recent experiment runs recorded in `~/.kagglex/runs.json`.

```bash
kagglex list [--limit N] [--credit]
```

### Options

| Flag        | Type    | Default | Description                              |
| ----------- | ------- | ------- | ---------------------------------------- |
| `--limit N` | Integer | `20`    | Maximum number of run records to display |
| `--credit`  | Flag    | False   | Show accounts and estimated quota instead (same as `kagglex account list`) |

## kagglex cancel

Cancel an active Kaggle kernel.

```bash
kagglex cancel <KERNEL_ID_OR_SLUG> [--account NAME]
```

`status`, `logs`, `pull` and `cancel` automatically use the account that submitted the run. Use `--account NAME` to override.

## kagglex dataset push

Create or update a private or public Kaggle dataset.

```bash
kagglex dataset push [OPTIONS]
```

### Options

| Flag            | Type   | Default        | Description                          |
| --------------- | ------ | -------------- | ------------------------------------ |
| `--dir PATH`    | Path   | Required       | Directory containing files to upload |
| `--title TITLE` | String | Required       | Dataset title                        |
| `--slug SLUG`   | String | Auto-generated | Dataset slug                         |
| `--public`      | Flag   | False          | Make dataset publicly accessible     |
| `--account NAME` | String | Default account | Account that will own the dataset |

## kagglex account

Manage named Kaggle accounts stored in `~/.kagglex/accounts.toml`. See [Multiple Kaggle Accounts](user-guide/accounts.md).

| Command | Description |
| ------- | ----------- |
| `kagglex account add NAME --username USER [--key KEY \| --token TOKEN \| --json PATH] [--default] [--force] [--verify]` | Register an account. Prompts for the key/token with hidden input if none is given |
| `kagglex account list [--days N] [--gpu-limit H] [--tpu-limit H]` | Show accounts with estimated GPU/TPU used and remaining |
| `kagglex account test [NAME ...]` | Contact Kaggle with each account (default: all) and report OK/FAIL |
| `kagglex account use NAME` | Set the default account |
| `kagglex account remove NAME` | Delete a stored account |
