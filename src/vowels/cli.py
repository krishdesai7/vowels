from typing import Annotated

import typer

from . import (
    Gender,
    Mode,
    detect_silences,
    extract_formants,
    label_textgrid,
    make_nucleus_points,
    save_chart,
)

app = typer.Typer(no_args_is_help=True, help="Vowel formant analysis toolkit.")


@app.command()
def silences(
    session: str,
    min_sounding_interval: Annotated[
        float, typer.Option(help="Minimum sounding interval (s)")
    ] = 0.1,
    min_silent_interval: Annotated[
        float, typer.Option(help="Minimum silent interval (s)")
    ] = 0.1,
    silence_threshold: Annotated[
        float, typer.Option(help="Silence threshold (dB)")
    ] = -25.0,
    min_pitch: Annotated[float, typer.Option(help="Minimum pitch (Hz)")] = 100.0,
) -> None:
    """Detect silences in a session WAV and write a TextGrid."""
    detect_silences(
        session,
        min_pitch=min_pitch,
        silence_threshold=silence_threshold,
        min_silent_interval=min_silent_interval,
        min_sounding_interval=min_sounding_interval,
    )


@app.command()
def label(
    session: str,
    labels_file: Annotated[str | None, typer.Option(help="Path to labels.txt")] = None,
) -> None:
    """Label sounding intervals in the TextGrid from labels.txt."""

    label_textgrid(session, labels_file=labels_file)


@app.command()
def nucleus(
    session: str,
    mode: Annotated[
        Mode,
        typer.Option(
            help="Mode (mono/diph/all)",
        ),
    ] = Mode.MONO,
) -> None:
    """Create nucleus point tier for formant extraction."""
    make_nucleus_points(session, mode)


@app.command()
def formants(
    session: str,
    gender: Annotated[Gender, typer.Option(help="Speaker gender (M/F/C)")] = Gender.M,
) -> None:
    """Extract F1/F2/F3 at nucleus points and write formants CSV."""
    extract_formants(session, gender)


@app.command()
def plot(
    session: str,
    mode: Annotated[Mode, typer.Option(help="Mode (mono/diph/all)")] = Mode.MONO,
) -> None:
    """Generate interactive vowel space HTML from existing formants CSV."""
    save_chart(session, mode)


@app.command()
def run(
    session: str,
    gender: Annotated[Gender, typer.Option(help="Speaker gender (M/F/C)")] = Gender.M,
    mode: Annotated[Mode, typer.Option(help="Mode (mono/diph/all)")] = Mode.MONO,
    min_sounding_interval: Annotated[
        float, typer.Option(help="Minimum sounding interval (s)")
    ] = 0.12,
) -> None:
    """Run the full pipeline: silences → label → nucleus → formants → plot."""
    detect_silences(session, min_sounding_interval)
    label_textgrid(session)
    make_nucleus_points(session, mode)
    extract_formants(session, gender)
    save_chart(session, mode)
