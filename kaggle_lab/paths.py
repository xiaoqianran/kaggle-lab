"""Repo-root discovery. Labs never hard-code ``../..``."""

from __future__ import annotations

from pathlib import Path


def find_repo_root(start: Path | None = None) -> Path:
    """Walk upward until this repo's markers appear."""
    here = (start or Path(__file__).resolve()).resolve()
    if here.is_file():
        here = here.parent
    for p in (here, *here.parents):
        has_pkg = (p / "kaggle_lab" / "__init__.py").is_file()
        has_marker = (p / "pyproject.toml").is_file()
        if has_pkg and has_marker:
            return p
    return Path(__file__).resolve().parents[1]


def ensure_import_path(root: Path | None = None) -> Path:
    """Put repo root on ``sys.path`` so labs can ``import kaggle_lab``."""
    import sys

    root = root or find_repo_root()
    text = str(root)
    if text not in sys.path:
        sys.path.insert(0, text)
    return root


REPO_ROOT = find_repo_root()
LABS_DIR = REPO_ROOT / "labs"
TRACKS_DIR = REPO_ROOT / "tracks"
DOCS_DIR = REPO_ROOT / "docs"
LOGS_DIR = REPO_ROOT / "logs"
ENV_MODEL_PROXY = REPO_ROOT / ".env.model-proxy"
