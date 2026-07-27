import fire
import parselmouth
from parselmouth.praat import call
from pathlib import Path


def detect_silences(session: str,
                    min_pitch: float = 100.0,
                    time_step: float = 0.0,
                    silence_threshold: float = -25.0,
                    min_silent_interval: float = 0.1,
                    min_sounding_interval: float = 0.1,
                    silent_label: str = "silent",
                    sounding_label: str = "sounding") -> None:
    dir: Path = Path(__file__).parent / "sessions" / session
    wav_path: Path = dir / f"{session}.wav"
    out_path: Path = dir / f"{session}.TextGrid"

    sound: parselmouth.Sound = parselmouth.Sound(str(wav_path))
    tg: parselmouth.Data = call(sound, "To TextGrid (silences)",
                                min_pitch,
                                time_step,
                                silence_threshold,
                                min_silent_interval,
                                min_sounding_interval,
                                silent_label,
                                sounding_label)
    call(tg, "Write to text file", str(out_path))
    print(f"Created {out_path}")


def main() -> None:
    fire.Fire(detect_silences)


if __name__ == "__main__":
    main()
