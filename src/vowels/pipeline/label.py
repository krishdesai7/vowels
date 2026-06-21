import csv
import sys
from pathlib import Path

import parselmouth

from ..paths import project_root, session_dir


def label_textgrid(session: str, labels_file: str | None = None) -> None:
    d: Path = session_dir(session)
    tg_path: Path = d / f"{session}.TextGrid"
    out_path: Path = d / f"{session}_labeled.TextGrid"
    intervals_csv: Path = d / f"{session}_intervals.csv"

    labels_path: Path = _resolve_labels_file(labels_file, d)
    with open(labels_path, encoding="utf-8") as f:
        labels: list[str] = [ln.strip() for ln in f if ln.strip()]

    tg: parselmouth.TextGrid = parselmouth.read(tg_path.as_posix())
    tier_number: int = 1
    n_intervals: int = parselmouth.praat.call(
        tg, "Get number of intervals", tier_number
    )

    sounding_indices: list[int] = [
        i
        for i in range(1, n_intervals + 1)
        if parselmouth.praat.call(tg, "Get label of interval", tier_number, i)
        == "sounding"
    ]

    if len(sounding_indices) == len(labels):
        for idx, interval_i in enumerate(sounding_indices):
            parselmouth.praat.call(
                tg, "Set interval text", tier_number, interval_i, labels[idx]
            )
        parselmouth.praat.call(tg, "Write to text file", out_path.as_posix())
        print(f"Created {out_path}")
        return

    if intervals_csv.exists():
        _label_from_csv(tg, tier_number, intervals_csv, out_path)
        return

    _write_diagnostic_csv(tg, tier_number, n_intervals, labels, intervals_csv)
    print(
        f"\nMismatch: detected {len(sounding_indices)} sounding intervals, "
        f"expected {len(labels)} labels.\n"
        f"Diagnostic CSV written to: {intervals_csv}\n"
        f"Edit the 'expected_label' column to fix mismatched rows, "
        f"then re-run `uv run vowels label {session}`.",
        file=sys.stderr,
    )
    raise SystemExit(1)


def _resolve_labels_file(labels_file: str | None, session_d: Path) -> Path:
    if labels_file:
        return Path(labels_file)
    if (session_d / "labels.txt").exists():
        return session_d / "labels.txt"
    return project_root() / "labels.txt"


def _write_diagnostic_csv(
    tg, tier_number: int, n_intervals: int, labels: list[str], out_path: Path
) -> None:
    label_idx = 0
    rows: list[dict[str, int | float | str | None]] = []
    for i in range(1, n_intervals + 1):
        lbl: str | None = parselmouth.praat.call(
            tg, "Get label of interval", tier_number, i
        )
        t1: float = parselmouth.praat.call(
            tg, "Get start time of interval", tier_number, i
        )
        t2: float = parselmouth.praat.call(
            tg, "Get end time of interval", tier_number, i
        )
        expected: str = ""
        if lbl == "sounding":
            expected = labels[label_idx] if label_idx < len(labels) else ""
            label_idx += 1
        rows.append(
            {
                "index": i,
                "t_start": round(t1, 6),
                "t_end": round(t2, 6),
                "duration": round(t2 - t1, 6),
                "detected_label": lbl,
                "expected_label": expected,
            }
        )

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer: csv.DictWriter[str] = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _label_from_csv(tg, tier_number: int, csv_path: Path, out_path: Path) -> None:
    with open(csv_path, newline="", encoding="utf-8") as f:
        rows: list[dict[str, str]] = list(csv.DictReader(f))

    for row in rows:
        if row["detected_label"] != "sounding":
            continue
        expected: str = row["expected_label"].strip()
        if expected:
            parselmouth.praat.call(
                tg, "Set interval text", tier_number, int(row["index"]), expected
            )

    parselmouth.praat.call(tg, "Write to text file", out_path.as_posix())
    print(f"Created {out_path} (from corrected CSV)")
