from pathlib import Path


def project_root() -> Path:
    p: Path = Path(__file__).resolve()
    for parent in p.parents:
        if (parent / "pyproject.toml").exists():
            return parent
    raise RuntimeError("Cannot find project root: no pyproject.toml found")


def session_dir(session: str) -> Path:
    return project_root() / "sessions" / session


data_dir: Path = project_root() / "data"

# IPA reference vowel positions overlaid on the plots: Hz (F0-F3) plus the Bark
# dimensions (Openness/Frontness/Roundness), 31 rows.
standards_file: Path = data_dir / "standards" / "male_standard_all.parquet"


def labels_file(session: str) -> Path:
    d: Path = session_dir(session)
    if (d / "labels.csv").exists():
        return d / "labels.csv"
    return data_dir / "labels.csv"
