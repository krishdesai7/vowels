import parselmouth
import fire
from parselmouth.praat import call
from pathlib import Path


def label_textgrid(session: str, labels_file: str = None) -> None:
    dir: Path = Path(__file__).parent / "sessions" / session
    tg_path: Path        = dir / f"{session}.TextGrid"
    labels_path: Path    = Path(labels_file) if labels_file else dir / "labels.txt"
    out_path: Path       = dir / f"{session}_labeled.TextGrid"

    with open(labels_path, "r", encoding="utf-8") as f:
        labels: list[str] = [ln.strip() for ln in f if ln.strip()]

    tg: parselmouth.Data = parselmouth.read(str(tg_path))

    tier_number: int = 1
    n_intervals: int = call(tg, "Get number of intervals", tier_number)

    sounding_indices: list[int] = [
        i for i in range(1, n_intervals + 1)
        if call(tg, "Get label of interval", tier_number, i) == "sounding"
    ]

    if len(sounding_indices) != len(labels):
        raise ValueError(
            f"Label count ({len(labels)}) ≠ sounding intervals ({len(sounding_indices)}). "
            "Fix the mismatch (merge/split intervals or adjust the label list)."
        )

    for idx, interval_i in enumerate[int](sounding_indices):
        call(tg, "Set interval text", tier_number, interval_i, labels[idx])

    call(tg, "Write to text file", str(out_path))

def main() -> None:
    fire.Fire(label_textgrid)


if __name__ == "__main__":
    main()