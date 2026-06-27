from collections.abc import Callable
from pathlib import Path


def project_root() -> Path:
    p: Path = Path(__file__).resolve()
    for parent in p.parents:
        if (parent / "pyproject.toml").exists():
            return parent
    raise RuntimeError("Cannot find project root: no pyproject.toml found")


session_dir: Callable[[str], Path] = lambda session: (  # noqa: E731
    project_root() / "sessions" / session
)
