"""Shared pytest fixtures and mocked clients for kagglex tests."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture(autouse=True)
def isolated_home(tmp_path_factory, monkeypatch):
    """Never touch the real ~/.kagglex (run history, accounts.toml) or Kaggle creds."""
    home = tmp_path_factory.mktemp("home")
    monkeypatch.setenv("HOME", str(home))
    for var in ("KAGGLE_API_TOKEN", "KAGGLE_USERNAME", "KAGGLE_KEY"):
        monkeypatch.delenv(var, raising=False)

    from kagglex.accounts import set_active_account

    set_active_account(None)
    yield home
    set_active_account(None)


@pytest.fixture
def mock_kaggle_api():
    """Mock the kaggle API client globally for offline tests."""
    with (
        patch("kagglex.client.get_kaggle_api") as mock_get_api,
        patch("kagglex.dataset.get_kaggle_api") as mock_get_dataset_api,
    ):
        api_instance = MagicMock()
        api_instance.get_config_value.return_value = "testuser"
        api_instance.config_values = {"username": "testuser"}

        # Push mock
        push_resp = MagicMock()
        push_resp.url = None
        push_resp.error = None
        api_instance.kernels_push.return_value = push_resp

        # Status mock
        status_resp = MagicMock()
        status_resp.status = "complete"
        status_resp.failure_message = None
        api_instance.kernels_status.return_value = status_resp

        # Cancel mock
        api_instance.kernels_cancel.return_value = True

        mock_get_api.return_value = api_instance
        mock_get_dataset_api.return_value = api_instance

        yield api_instance


@pytest.fixture
def sample_script(tmp_path: Path) -> Path:
    """Create a sample standalone Python script."""
    script_file = tmp_path / "train.py"
    script_file.write_text(
        "import sys\nprint('Training model...')\n",
        encoding="utf-8",
    )
    return script_file


@pytest.fixture
def sample_src_project(tmp_path: Path) -> Path:
    """Create a sample src/ layout project."""
    proj_dir = tmp_path / "my_ml_project"
    proj_dir.mkdir(parents=True)
    (proj_dir / "pyproject.toml").write_text(
        '[project]\nname = "my_ml_project"\nversion = "0.1.0"\n',
        encoding="utf-8",
    )
    src_dir = proj_dir / "src" / "my_ml_project"
    src_dir.mkdir(parents=True)
    (src_dir / "__init__.py").write_text('__version__ = "0.1.0"\n', encoding="utf-8")
    (src_dir / "train.py").write_text("def run(): pass\n", encoding="utf-8")

    # Add ignored junk
    cache_dir = src_dir / "__pycache__"
    cache_dir.mkdir()
    (cache_dir / "train.cpython-312.pyc").write_text("cache", encoding="utf-8")

    return proj_dir
