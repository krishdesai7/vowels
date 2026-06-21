from pathlib import Path

import parselmouth

from ..paths import session_dir


def detect_silences(
    session: str,
    min_pitch: float = 100.0,
    time_step: float = 0.0,
    silence_threshold: float = -25.0,
    min_silent_interval: float = 0.1,
    min_sounding_interval: float = 0.1,
    silent_label: str = "silent",
    sounding_label: str = "sounding",
) -> None:
    d: Path = session_dir(session)
    wav_path: Path = d / f"{session}.wav"
    out_path: Path = d / f"{session}.TextGrid"

    sound: parselmouth.Sound = parselmouth.Sound(str(wav_path))
    tg: parselmouth.TextGrid = parselmouth.praat.call(
        sound,
        "To TextGrid (silences)",
        min_pitch,
        time_step,
        silence_threshold,
        min_silent_interval,
        min_sounding_interval,
        silent_label,
        sounding_label,
    )
    parselmouth.praat.call(tg, "Write to text file", str(out_path))
    print(f"Created {out_path}")
