import csv
from typing import TYPE_CHECKING

import parselmouth

from ..labels import read_labels
from ..paths import labels_file, session_dir

if TYPE_CHECKING:
    from pathlib import Path


def label_textgrid(session: str) -> None:
    d: Path = session_dir(session)
    tg_path: Path = d / f"{session}.TextGrid"
    out_path: Path = d / f"{session}_labeled.TextGrid"
    intervals_csv: Path = d / f"{session}_intervals.csv"

    labels: list[str] = read_labels(labels_file(session))

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
        return

    if intervals_csv.exists():
        _label_from_csv(tg, tier_number, intervals_csv, out_path)
        return

    _write_diagnostic_csv(tg, tier_number, n_intervals, labels, intervals_csv)
    raise SystemExit(1)


def _write_diagnostic_csv(
    tg: parselmouth.TextGrid,
    tier_number: int,
    n_intervals: int,
    labels: list[str],
    out_path: Path,
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

    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer: csv.DictWriter[str] = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _label_from_csv(
    tg: parselmouth.TextGrid, tier_number: int, csv_path: Path, out_path: Path
) -> None:
    with csv_path.open(newline="", encoding="utf-8") as f:
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
