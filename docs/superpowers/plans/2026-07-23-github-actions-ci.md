# GitHub Actions CI Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a single GitHub Actions workflow that runs format check, ruff lint, pyrefly typecheck, and pytest on push/PR to `main`.

**Architecture:** One workflow file under `.github/workflows/ci.yml`. Uses `astral-sh/setup-uv` with cache, syncs the locked env with the `dev` group, then runs four check-only commands in sequence. No local task runner.

**Tech Stack:** GitHub Actions, uv, ruff (via `uv format` / `uv run ruff`), pyrefly, pytest

## Global Constraints

- Triggers: `push` and `pull_request` to `main` only
- CI commands must be non-mutating: `uv format --check`, `uv run ruff check` (no `--fix`)
- Both `uv run ruff check` and `uv run pyrefly check` are required (lint vs typecheck); do not use `uv check` (ty)
- Single Python version (project `requires-python = ">=3.14"`); no matrix
- Install with `uv sync --locked --group dev`
- Pin `astral-sh/setup-uv` by commit SHA; pin uv version (e.g. `0.11.30`)
- Do not commit unless the user explicitly asks

---

## File Structure

| File | Responsibility |
|------|----------------|
| `.github/workflows/ci.yml` | Sole deliverable — CI job definition |
| `docs/superpowers/specs/2026-07-23-github-actions-ci-design.md` | Already written; no changes required |
| `docs/superpowers/plans/2026-07-23-github-actions-ci.md` | This plan |

---

### Task 1: Add CI workflow

**Files:**
- Create: `.github/workflows/ci.yml`

**Interfaces:**
- Consumes: `uv.lock`, `pyproject.toml` `dev` dependency group, existing tests under `tests/`
- Produces: GitHub Actions `check` job that fails on any of the four gates

- [x] **Step 1: Create the workflow file**

Create `.github/workflows/ci.yml` with this content (adjust `setup-uv` commit SHA to the current pinned release if needed; keep `version: "0.11.30"` unless bumping intentionally):

```yaml
name: CI

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  check:
    name: check
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v7

      - name: Install uv
        uses: astral-sh/setup-uv@08807647e7069bb48b6ef5acd8ec9567f424441b # v8.1.0
        with:
          version: "0.11.30"
          enable-cache: true

      - name: Install the project
        run: uv sync --locked --group dev

      - name: Format
        run: uv format --check

      - name: Lint
        run: uv run ruff check

      - name: Typecheck
        run: uv run pyrefly check

      - name: Test
        run: uv run pytest -q
```

- [x] **Step 2: Validate YAML locally (syntax / commands)**

Run from repo root:

```bash
# Confirm the same commands succeed locally (optional but preferred)
uv sync --locked --group dev
uv format --check
uv run ruff check
uv run pyrefly check
uv run pytest -q
```

Expected: all exit 0 on a clean tree. If format/lint fail due to pre-existing issues, fix only what is required for CI to be useful, or leave as known failures for a follow-up — do not drive-by refactor.

Also confirm the workflow file parses as YAML:

```bash
python -c "import pathlib, yaml; yaml.safe_load(pathlib.Path('.github/workflows/ci.yml').read_text())"
```

Expected: no exception. If PyYAML is unavailable:

```bash
uv run python -c "import pathlib; p=pathlib.Path('.github/workflows/ci.yml'); assert p.exists() and 'uv format --check' in p.read_text() and 'uv run ruff check' in p.read_text() and 'uv run pyrefly check' in p.read_text() and 'uv run pytest -q' in p.read_text()"
```

- [ ] **Step 3: Commit (only if the user asks)**

```bash
git add .github/workflows/ci.yml docs/superpowers/specs/2026-07-23-github-actions-ci-design.md docs/superpowers/plans/2026-07-23-github-actions-ci.md
git commit -m "$(cat <<'EOF'
Add GitHub Actions CI for format, lint, typecheck, and tests.

EOF
)"
```

Skip this step unless the user explicitly requests a commit.
