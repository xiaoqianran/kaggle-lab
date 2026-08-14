from __future__ import annotations

from kaggle_lab.paths import (
    DATA_DIR,
    ENV_MODEL_PROXY,
    LABS_DIR,
    REPO_ROOT,
    find_repo_root,
)


def test_find_repo_root_from_nested_lab() -> None:
    nested = LABS_DIR / "001-model-proxy" / "run.py"
    assert nested.is_file()
    assert find_repo_root(nested) == REPO_ROOT


def test_env_file_stays_at_repo_root() -> None:
    assert ENV_MODEL_PROXY == REPO_ROOT / ".env.model-proxy"


def test_model_table_lives_in_data() -> None:
    assert (DATA_DIR / "kaggle_ai_models.json").is_file()
