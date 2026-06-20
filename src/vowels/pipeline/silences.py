import parselmouth
from parselmouth.praat import call

from vowels.paths import session_dir


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
    d = session_dir(session)
    wav_path = d / f"{session}.wav"
    out_path = d / f"{session}.TextGrid"

    sound = parselmouth.Sound(str(wav_path))
    tg = call(
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
    call(tg, "Write to text file", str(out_path))
    print(f"Created {out_path}")
