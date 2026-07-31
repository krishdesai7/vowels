# Justfile

# Run mutable development fixes (modifies code)
# uv run just m[utable]
alias m := mutable
mutable:
    uv sync
    uv run pyrefly infer --return-types --parameter-types --imports --containers
    uv format
    uv run ruff check --fix --unsafe-fixes
    uv run pyrefly check
    uv run pytest -q

# Run immutable repository checks (read-only)
# uv run just i[mmutable]
alias i := immutable
immutable:
    #!/usr/bin/env bash
    set -euo pipefail
    export UV_LOCKED=1

    uv sync
    uv run pyrefly check
    uv format --diff
    uv run ruff check
    uv audit
    uv run complexipy
    uv run pytest -q

upgrade:
    uv lock --upgrade
    uv sync