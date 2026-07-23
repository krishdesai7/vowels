# GitHub Actions CI design

Date: 2026-07-23

## Goal

Add a single GitHub Actions workflow that gates PRs/pushes with format, lint, typecheck, and tests. No local task runner or pre-commit in this scope.

## Triggers

- `push` to `main`
- `pull_request` targeting `main`

## Workflow

**Path:** `.github/workflows/ci.yml`  
**Job:** one `check` job on `ubuntu-latest`

Steps (aligned with [uv’s GitHub Actions guide](https://docs.astral.sh/uv/guides/integration/github/#syncing-and-running)):

1. `actions/checkout`
2. `astral-sh/setup-uv` with `enable-cache: true` and uv version pinned (e.g. `0.11.30`)
3. `uv sync --locked --group dev` — install from `uv.lock`, including ruff, pyrefly, and pytest
4. Quality gates, in order:

```bash
uv format --check
uv run ruff check
uv run pyrefly check
uv run pytest -q
```

## CI vs local commands

| Local (mutating OK)       | CI (check-only)         |
| ------------------------- | ----------------------- |
| `uv format`               | `uv format --check`     |
| `uv run ruff check --fix` | `uv run ruff check`     |
| `uv run pyrefly check`    | `uv run pyrefly check`  |
| `uv run pytest -q`        | `uv run pytest -q`      |

Both `ruff check` and `pyrefly check` are required: ruff lints; pyrefly typechecks. Prefer pyrefly over `uv check` (ty) given maturity.

## Out of scope

- Python version matrix (project requires `>=3.14`; one runner version is enough)
- Makefile / just / pre-commit
- Publishing or release workflows

## Success criteria

- Workflow file is present and runnable on GitHub Actions
- A PR that fails format, lint, typecheck, or tests fails the `check` job
- A clean tree passes all four steps
