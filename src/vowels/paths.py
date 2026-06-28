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

data_dir: Path = project_root() / "data"


def labels_file(session: str) -> Path:
    d: Path = session_dir(session)
    if (d / "labels.txt").exists():
        return d / "labels.txt"
    return data_dir / "labels.txt"
