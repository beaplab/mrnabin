import argparse
import sys
from typing import Any, Callable
from unittest.mock import patch

import pytest

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


_LAYERS: dict[str, dict[str, Any]] = {
    "base.yaml": {"k": 1, "shared": "from_base"},
    "overlay.yaml": {"k": 2},
    "custom.yaml": {"k": 3},
}


@pytest.mark.parametrize(
    "argv, expected",
    [
        ([], 2),  # later base config over earlier
        (["--config", "custom.yaml"], 3),  # --config over base configs
        (["--k", "99"], 99),  # CLI flag over base configs
        (["--config", "custom.yaml", "--k", "99"], 99),  # CLI flag over --config
    ],
)
def test_precedence(argv: list[str], expected: int) -> None:
    args = _run(argv, _LAYERS, ["base.yaml", "overlay.yaml"])
    assert args.k == expected
    assert args.shared == "from_base"


def test_caller_parser_args_preserved() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--extra", default="default_extra")
    args = _run(["--extra", "passed"], {"base.yaml": {"k": 5}}, ["base.yaml"], parser=parser)
    assert args.extra == "passed"
    assert args.k == 5


@pytest.mark.parametrize("default, flip", [(True, "--no-flag"), (False, "--flag")])
def test_bool_default_is_respected(default: bool, flip: str) -> None:
    files = {"base.yaml": {"flag": default}}
    assert _run([], files, ["base.yaml"]).flag is default
    assert _run([flip], files, ["base.yaml"]).flag is not default


def test_list_type_default_and_override() -> None:
    files = {"base.yaml": {"items": ["a", "b"]}}
    args = _run([], files, ["base.yaml"])
    assert args.items == ["a", "b"]
    args = _run(["--items", "x", "y", "z"], files, ["base.yaml"])
    assert args.items == ["x", "y", "z"]
