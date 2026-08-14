from __future__ import annotations

import json

from kaggle_lab.catalog import (
    JOURNEYS,
    LABS,
    TRACKS,
    discover_unregistered_labs,
    discover_unregistered_tracks,
    resolve_lab,
    resolve_track,
)
from kaggle_lab.cli import journey_argv, main
from kaggle_lab.paths import LABS_DIR, REPO_ROOT, TRACKS_DIR


def test_repo_root_looks_right() -> None:
    assert (REPO_ROOT / "kaggle_lab" / "catalog.py").is_file()
    assert LABS_DIR.is_dir()
    assert TRACKS_DIR.is_dir()


def test_every_catalog_lab_has_run_py() -> None:
    missing = [lab.id for lab in LABS if not lab.run_py.is_file()]
    assert missing == [], f"catalog labs missing run.py: {missing}"


def test_every_catalog_track_exists() -> None:
    missing = [tr.id for tr in TRACKS if not tr.dir.is_dir()]
    assert missing == [], f"catalog tracks missing dir: {missing}"


def test_no_unregistered_labs_or_tracks() -> None:
    assert discover_unregistered_labs() == []
    assert discover_unregistered_tracks() == []


def test_resolve_lab_aliases() -> None:
    assert resolve_lab("001").id == "001-model-proxy"
    assert resolve_lab("014").id == "014-camel-workforce-bench"
    assert resolve_lab("workforce").id == "014-camel-workforce-bench"
    assert resolve_lab("015-dual-agent-chat").id == "015-dual-agent-chat"
    assert resolve_lab("quota").id == "010-quota-dashboard"
    assert resolve_lab("does-not-exist") is None


def test_resolve_track_aliases() -> None:
    assert resolve_track("v03").id == "image-classification"
    assert resolve_track("arc").id == "arc-agi-3"
    assert resolve_track("depth-estimation").id == "depth-estimation"
    assert resolve_track("nope") is None


def test_lab_aliases_are_unique() -> None:
    seen: dict[str, str] = {}
    clashes: list[str] = []
    for lab in LABS:
        for key in (lab.id, *lab.aliases):
            key = key.lower()
            if key in seen:
                clashes.append(f"{key}: {seen[key]} vs {lab.id}")
            else:
                seen[key] = lab.id
    assert clashes == [], f"duplicate lab aliases: {clashes}"


def test_cli_list_json(capsys) -> None:
    assert main(["list", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert len(payload["labs"]) == len(LABS)
    assert len(payload["tracks"]) == len(TRACKS)
    ids = {row["id"] for row in payload["labs"]}
    assert "015-dual-agent-chat" in ids


def test_cli_help_is_a_map(capsys) -> None:
    assert main([]) == 0
    out = capsys.readouterr().out
    assert "你想做什么" in out
    assert "labs/" in out
    assert "tracks/" in out


def test_journey_argv_inject() -> None:
    by_id = {j.id: j for j in JOURNEYS}
    assert journey_argv(by_id["chat"], ["你好"]) == ["chat", "你好"]
    assert journey_argv(by_id["auth"], []) == ["auth"]
    assert journey_argv(by_id["workforce"], []) == ["run"]
    assert journey_argv(by_id["workforce"], ["--theme", "x"]) == ["run", "--theme", "x"]
    assert journey_argv(by_id["workforce"], ["show"]) == ["show"]
    assert journey_argv(by_id["quota"], ["usage"]) == ["usage"]
    assert journey_argv(by_id["debate"], ["--rounds", "3"]) == ["run", "--rounds", "3"]


def test_cli_unknown() -> None:
    assert main(["definitely-not-a-lab"]) == 2


def test_cli_track_unknown() -> None:
    assert main(["track", "no-such-track"]) == 2


def test_cli_track_list(capsys) -> None:
    assert main(["track"]) == 0
    out = capsys.readouterr().out
    assert "arc-agi-3" in out
    assert "image-classification" in out


def test_sae_help_dispatches() -> None:
    try:
        code = main(["004", "--help"])
    except SystemExit as exc:
        code = int(exc.code or 0)
    assert code == 0


def test_mcp_lab_still_importable_by_siblings() -> None:
    assert (LABS_DIR / "007-mcp-harness" / "run.py").is_file()
    assert (LABS_DIR / "004-sae").is_dir()
