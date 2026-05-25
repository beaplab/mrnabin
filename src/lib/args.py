import argparse
from typing import Any

import yaml


def _parser_w_defaults(
    config: dict[str, Any], parser: argparse.ArgumentParser | None = None
) -> argparse.ArgumentParser:
    if parser is None:
        parser = argparse.ArgumentParser()
    for k, v in config.items():
        if type(v) is bool:
            parser.add_argument(f"--{k}", default=v, action=argparse.BooleanOptionalAction)
        elif type(v) is list:
            parser.add_argument(f"--{k}", default=v, nargs="+")
        else:
            parser.add_argument(f"--{k}", default=v, type=type(v))
    parser.add_argument("-c", "--config", default=None, type=str)
    return parser


def _load_yaml(path: str) -> dict[str, Any]:
    with open(path) as f:
        return yaml.safe_load(f) or {}


def get_args(config_paths: list[str], parser: argparse.ArgumentParser | None = None) -> argparse.Namespace:
    """Precedence (highest -> lowest):
    CLI flags > --config file > later config_paths > earlier config_paths.
    """
    peek = argparse.ArgumentParser(add_help=False)
    peek.add_argument("-c", "--config", default=None)
    peek_args, _ = peek.parse_known_args()

    config: dict[str, Any] = {}
    for path in config_paths:
        config.update(_load_yaml(path))
    if peek_args.config:
        config.update(_load_yaml(peek_args.config))

    parser = _parser_w_defaults(config, parser)
    return parser.parse_args()
