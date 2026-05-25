import argparse
import sys
from typing import Any, Callable
from unittest.mock import patch

from lib.args import get_args


def _yaml_loader(files: dict[str, dict[str, Any]]) -> Callable[[str], dict[str, Any]]:
    def _load(path: str) -> dict[str, Any]:
        return files[path]

    return _load


def _run(
    argv: list[str],
    files: dict[str, dict[str, Any]],
    config_paths: list[str],
    parser: argparse.ArgumentParser | None = None,
) -> argparse.Namespace:
    with patch.object(sys, "argv", ["prog"] + argv), patch("lib.args._load_yaml", side_effect=_yaml_loader(files)):
        return get_args(config_paths, parser)


def test_single_base_config_defaults() -> None:
    args = _run([], {"base.yaml": {"k": 5, "name": "foo"}}, ["base.yaml"])
    assert args.k == 5
    assert args.name == "foo"


def test_later_config_overrides_earlier() -> None:
    files: dict[str, dict[str, Any]] = {"base.yaml": {"k": 1, "shared": "from_base"}, "overlay.yaml": {"k": 2}}
    args = _run([], files, ["base.yaml", "overlay.yaml"])
    assert args.k == 2
    assert args.shared == "from_base"


def test_cli_flag_overrides_base_config() -> None:
    args = _run(["--k", "99"], {"base.yaml": {"k": 5}}, ["base.yaml"])
    assert args.k == 99


def test_custom_config_overrides_base() -> None:
    files = {"base.yaml": {"k": 5}, "custom.yaml": {"k": 50}}
    args = _run(["--config", "custom.yaml"], files, ["base.yaml"])
    assert args.k == 50


def test_cli_flag_overrides_custom_config() -> None:
    files = {"base.yaml": {"k": 5}, "custom.yaml": {"k": 50}}
    args = _run(["--config", "custom.yaml", "--k", "99"], files, ["base.yaml"])
    assert args.k == 99


def test_caller_parser_args_preserved() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--extra", default="default_extra")
    args = _run(["--extra", "passed"], {"base.yaml": {"k": 5}}, ["base.yaml"], parser=parser)
    assert args.extra == "passed"
    assert args.k == 5


def test_bool_default_true_is_respected() -> None:
    files = {"base.yaml": {"flag": True}}
    assert _run([], files, ["base.yaml"]).flag is True
    assert _run(["--no-flag"], files, ["base.yaml"]).flag is False


def test_bool_default_false_is_respected() -> None:
    files = {"base.yaml": {"flag": False}}
    assert _run([], files, ["base.yaml"]).flag is False
    assert _run(["--flag"], files, ["base.yaml"]).flag is True


def test_list_type_default_and_override() -> None:
    files = {"base.yaml": {"items": ["a", "b"]}}
    args = _run([], files, ["base.yaml"])
    assert args.items == ["a", "b"]
    args = _run(["--items", "x", "y", "z"], files, ["base.yaml"])
    assert args.items == ["x", "y", "z"]
