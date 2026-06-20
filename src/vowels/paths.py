from pathlib import Path


def project_root() -> Path:
    p = Path(__file__).resolve()
    for parent in p.parents:
        if (parent / "pyproject.toml").exists():
            return parent
    raise RuntimeError("Cannot find project root: no pyproject.toml found")


def session_dir(session: str) -> Path:
    return project_root() / "sessions" / session
