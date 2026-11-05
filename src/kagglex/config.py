"""Configuration models and validation rules for kagglex."""

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

SUPPORTED_ACCELERATORS: set[str] = {
    "t4-2x",
    "p100",
    "v3-8",
    "none",
}


def slugify(value: str) -> str:
    """Convert arbitrary string into a valid Kaggle slug.

    Kaggle slugs must be lowercase, alphanumeric with hyphens, and 5-50 chars.
    """
    slug = re.sub(r"[^\w\s-]", "", value.lower())
    slug = re.sub(r"[-\s]+", "-", slug).strip("-")
    if len(slug) < 5:
        slug = f"{slug}-exp"
    return slug[:50]


@dataclass
class RunConfig:
    """Configuration for a remote Kaggle experiment execution."""

    command: str
    title: str
    slug: str | None = None
    project_dir: Path | None = None
    target_file: Path | None = None
    gpu_type: str = "t4-2x"
    enable_tpu: bool = False
    multi_gpu: bool = False
    enable_internet: bool = True
    dataset_slugs: list[str] = field(default_factory=list)
    include_data: list[str] = field(default_factory=list)
    extra_pip_deps: list[str] = field(default_factory=list)
    env_vars: dict[str, str] = field(default_factory=dict)
    kaggle_secrets: list[str] = field(default_factory=list)
    parent_kernels: list[str] = field(default_factory=list)
    output_dir: Path | None = None
    record_file: Path | None = None
    include_outputs: list[str] = field(default_factory=list)
    exclude_outputs: list[str] = field(default_factory=list)
    auto_dataset: bool = False
    auto_dataset_slug: str | None = None
    poll_interval: int = 20
    timeout_sec: int = 43200
    # Named kagglex account ("Phong"), "auto", or None for the default account.
    account: str | None = None

    def __post_init__(self) -> None:
        """Validate and normalize configuration parameters."""
        if not self.command or not self.command.strip():
            raise ValueError("Command cannot be empty.")
        self.command = self.command.strip()

        if not self.title or not self.title.strip():
            raise ValueError("Title cannot be empty.")
        self.title = self.title.strip()

        if not self.slug:
            self.slug = slugify(self.title)
        else:
            self.slug = slugify(self.slug)

        if self.auto_dataset_slug:
            self.auto_dataset_slug = slugify(self.auto_dataset_slug)

        self.gpu_type = self.gpu_type.lower()
        if self.gpu_type not in SUPPORTED_ACCELERATORS:
            raise ValueError(
                f"Invalid gpu_type '{self.gpu_type}'. Must be one of {sorted(SUPPORTED_ACCELERATORS)}"
            )

        if self.gpu_type == "v3-8":
            self.enable_tpu = True

        if self.project_dir and isinstance(self.project_dir, str):
            self.project_dir = Path(self.project_dir).resolve()

        if self.target_file and isinstance(self.target_file, str):
            self.target_file = Path(self.target_file).resolve()

        if self.output_dir and isinstance(self.output_dir, str):
            self.output_dir = Path(self.output_dir).resolve()

        if self.record_file and isinstance(self.record_file, str):
            self.record_file = Path(self.record_file).resolve()


@dataclass
class DatasetConfig:
    """Configuration for Kaggle dataset creation and versioning."""

    title: str
    data_dir: Path
    slug: str | None = None
    is_public: bool = False
    license_name: str = "CC0-1.0"

    def __post_init__(self) -> None:
        """Validate and normalize dataset parameters."""
        if not self.title or not self.title.strip():
            raise ValueError("Dataset title cannot be empty.")
        self.title = self.title.strip()

        if not self.slug:
            self.slug = slugify(self.title)
        else:
            self.slug = slugify(self.slug)

        if isinstance(self.data_dir, str):
            self.data_dir = Path(self.data_dir).resolve()
        else:
            self.data_dir = self.data_dir.resolve()

        if not self.data_dir.exists():
            raise FileNotFoundError(
                f"Dataset directory does not exist: {self.data_dir}"
            )


@dataclass
class InteractiveConfig:
    """Configuration for Kaggle Jupyter proxy interactive sessions."""

    url: str
    timeout: int = 120
    verbose: bool = True


def read_toml_file(path: Path) -> dict[str, Any]:
    """Parse a TOML file safely using standard library tomllib or tomli."""
    if not path.is_file():
        return {}
    try:
        import tomllib
    except ImportError:
        try:
            import tomli as tomllib  # type: ignore[no-redef]
        except ImportError:
            logger.debug("Neither tomllib nor tomli is available to parse %s", path)
            return {}

    try:
        with open(path, "rb") as f:
            return tomllib.load(f)
    except Exception as e:
        logger.warning("Failed to parse TOML file %s: %s", path, e)
        return {}


def _merge_configs(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge two configuration dictionaries."""
    merged = dict(base)
    for k, v in override.items():
        if k in merged and isinstance(merged[k], dict) and isinstance(v, dict):
            merged[k] = _merge_configs(merged[k], v)
        else:
            merged[k] = v
    return merged


def get_global_config_file(custom_path: Path | None = None) -> Path | None:
    """Find the path to user global config file if it exists."""
    if custom_path and custom_path.is_file():
        return custom_path

    candidates = [
        Path.home() / ".kagglex" / "config.toml",
        Path.home() / ".kagglex" / "kagglex.toml",
        Path.home() / ".config" / "kagglex" / "config.toml",
    ]
    for p in candidates:
        if p.is_file():
            return p
    return None


def load_project_config(
    start_dir: Path | None = None,
    global_config_path: Path | None = None,
) -> dict[str, Any]:
    """Load configuration hierarchy from user global config, pyproject.toml, and kagglex.toml.

    Search hierarchy (later overrides earlier):
    1. User global config (~/.kagglex/config.toml, ~/.kagglex/kagglex.toml, ~/.config/kagglex/config.toml)
    2. pyproject.toml [tool.kagglex] in project / repo directory
    3. kagglex.toml in project / repo directory
    """
    merged: dict[str, Any] = {}

    # 1. User global configuration
    g_file = (
        global_config_path
        if (global_config_path and global_config_path.is_file())
        else get_global_config_file()
    )
    if g_file and g_file.is_file():
        g_data = read_toml_file(g_file)
        tool_kagglex = g_data.get("tool", {}).get("kagglex")
        if isinstance(tool_kagglex, dict):
            merged = _merge_configs(merged, tool_kagglex)
        elif g_data:
            merged = _merge_configs(merged, g_data)

    # 2. Project-level configuration
    proj_root = (start_dir or Path.cwd()).resolve()

    # Look in pyproject.toml
    pyproject_path = proj_root / "pyproject.toml"
    if not pyproject_path.is_file():
        for parent in proj_root.parents:
            if (parent / "pyproject.toml").is_file():
                pyproject_path = parent / "pyproject.toml"
                break

    if pyproject_path.is_file():
        pyproj_data = read_toml_file(pyproject_path)
        tool_kagglex = pyproj_data.get("tool", {}).get("kagglex")
        if isinstance(tool_kagglex, dict):
            merged = _merge_configs(merged, tool_kagglex)

    # Look in kagglex.toml
    kagglex_path = proj_root / "kagglex.toml"
    if not kagglex_path.is_file():
        for parent in proj_root.parents:
            if (parent / "kagglex.toml").is_file():
                kagglex_path = parent / "kagglex.toml"
                break

    if kagglex_path.is_file():
        kagglex_data = read_toml_file(kagglex_path)
        tool_data = kagglex_data.get("tool", {}).get("kagglex")
        if isinstance(tool_data, dict):
            merged = _merge_configs(merged, tool_data)
        elif kagglex_data:
            merged = _merge_configs(merged, kagglex_data)

    return merged


def resolve_jupyter_url(
    explicit_url: str | None = None,
    start_dir: Path | None = None,
    global_config_path: Path | None = None,
) -> str | None:
    """Resolve Kaggle Jupyter URL from parameter, KAGGLE_JUPYTER_URL env var, or config."""
    import os

    if explicit_url and explicit_url.strip():
        return explicit_url.strip()
    env_url = os.environ.get("KAGGLE_JUPYTER_URL", "").strip()
    if env_url:
        return env_url

    cfg = load_project_config(
        start_dir=start_dir, global_config_path=global_config_path
    )
    cfg_url = cfg.get("url") or cfg.get("jupyter_url")
    if cfg_url and isinstance(cfg_url, str) and cfg_url.strip():
        return cfg_url.strip()

    return None
