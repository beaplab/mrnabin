PHONY: back isort format black_check isort_check format_check mypy flake8 lint test install install_dev

TARGETS=src scripts
VENV_NAME = .venv

all: | pr install

pr: | format lint test

black:
	uv run black --config=pyproject.toml ${TARGETS}

isort:
	uv run isort --settings-path=pyproject.toml ${TARGETS}

format: | black isort

black_check:
	uv run black --config=pyproject.toml --check ${TARGETS}

isort_check:
	uv run isort --settings-path=pyproject.toml --check ${TARGETS}

format_check: | black_check isort_check

mypy:
	uv run mypy --config-file=pyproject.toml ${TARGETS}

flake8:
	uv run flake8 --config=.flake8 ${TARGETS}

lint: | mypy flake8

test:
	uv run pytest src

install:
	uv sync

install_dev:
	uv sync --extra dev

clean:
	rm uv.lock \
	&& rm .python-version \
	&& rm -rf .venv \
	&& rm -rf **/**/.mypy_cache \
	&& rm -rf **/**/*~ \
	&& rm -rf .pytest_cache \
	&& rm -rf **/**/.DS_Store \
	&& rm -rf **/**/__pycache__
