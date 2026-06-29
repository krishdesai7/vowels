from typing import Annotated

import typer

from . import (
    Gender,
    detect_silences,
    extract_formants,
    label_textgrid,
    save_bark_chart,
    save_bark_projections,
    save_chart,
)

app = typer.Typer(no_args_is_help=True, help="Vowel formant analysis toolkit.")


@app.command()
def silences(
    session: str,
    min_sounding_interval: Annotated[
        float,
        typer.Option(
            "--min-sounding-interval", "-s", help="Minimum sounding interval (s)"
        ),
    ] = 0.08,
    min_silent_interval: Annotated[
        float,
        typer.Option("--min-silent-interval", "-i", help="Minimum silent interval (s)"),
    ] = 0.1,
    silence_threshold: Annotated[
        float, typer.Option("--silence-threshold", "-t", help="Silence threshold (dB)")
    ] = -25.0,
    min_pitch: Annotated[
        float, typer.Option("--min-pitch", "-p", help="Minimum pitch (Hz)")
    ] = 100.0,
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
) -> None:
    """Label sounding intervals in the TextGrid from labels.txt."""
    label_textgrid(session)


@app.command()
def formants(
    session: str,
    gender: Annotated[
        Gender,
        typer.Option(
            "--gender",
            "-g",
            help="Speaker gender (M/F/C)",
            case_sensitive=False,
        ),
    ] = Gender.M,
) -> None:
    """Extract formant trajectories with fasttrackpy and write the trajectory parquet."""
    extract_formants(session, gender)


@app.command()
def plot(session: str) -> None:
    """Generate interactive vowel space HTML from the trajectory parquet."""
    save_chart(session)


@app.command()
def bark(session: str) -> None:
    """Generate interactive Bark Z 3D vowel space HTML from the trajectory parquet."""
    save_bark_chart(session)


@app.command()
def projections(session: str) -> None:
    """Generate three 2D Bark Z projection plots (Frontness×Openness, ×Roundness, Openness×Roundness)."""
    save_bark_projections(session)


@app.command()
def run(
    session: str,
    gender: Annotated[
        Gender,
        typer.Option(
            "--gender",
            "-g",
            help="Speaker gender (M/F/C)",
            case_sensitive=False,
        ),
    ] = Gender.M,
    min_sounding_interval: Annotated[
        float,
        typer.Option(
            "--min-sounding-interval", "-s", help="Minimum sounding interval (s)"
        ),
    ] = 0.08,
) -> None:
    """Run the full pipeline: silences → label → formants → plots."""
    detect_silences(session, min_sounding_interval=min_sounding_interval)
    label_textgrid(session)
    extract_formants(session, gender)
    save_chart(session)
    save_bark_chart(session)
    save_bark_projections(session)
