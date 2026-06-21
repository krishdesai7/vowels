from enum import Enum
from typing import Annotated

import typer

app = typer.Typer(no_args_is_help=True, help="Vowel formant analysis toolkit.")


class Gender(str, Enum):
    M = "M"
    F = "F"
    C = "C"


@app.command()
def silences(
    session: str,
    min_sounding_interval: Annotated[float, typer.Option(help="Minimum sounding interval (s)")] = 0.1,
    min_silent_interval: Annotated[float, typer.Option(help="Minimum silent interval (s)")] = 0.1,
    silence_threshold: Annotated[float, typer.Option(help="Silence threshold (dB)")] = -25.0,
    min_pitch: Annotated[float, typer.Option(help="Minimum pitch (Hz)")] = 100.0,
) -> None:
    """Detect silences in a session WAV and write a TextGrid."""
    from vowels.pipeline.silences import detect_silences

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
    from vowels.pipeline.label import label_textgrid

    label_textgrid(session, labels_file=labels_file)


@app.command()
def nucleus(
    session: str,
    diphthongs: Annotated[bool, typer.Option("--diphthongs/--no-diphthongs", help="Enable diphthong two-point measurement")] = True,
) -> None:
    """Create nucleus point tier for formant extraction."""
    from vowels.pipeline.nucleus import make_nucleus_points

    make_nucleus_points(session, diphthongs=diphthongs)


@app.command()
def formants(
    session: str,
    gender: Annotated[Gender, typer.Option(help="Speaker gender (M/F/C)")] = Gender.M,
) -> None:
    """Extract F1/F2/F3 at nucleus points and write formants CSV."""
    from vowels.pipeline.formants import extract_formants

    extract_formants(session, gender=gender.value)


@app.command()
def plot(
    session: str,
    diphthongs: Annotated[bool, typer.Option("--diphthongs", help="Show diphthongs only", is_flag=True)] = False,
    all_vowels: Annotated[bool, typer.Option("--all", help="Show all vowels", is_flag=True)] = False,
) -> None:
    """Generate interactive vowel space HTML from existing formants CSV."""
    from vowels.plots.vowel_space import save_chart

    if diphthongs and all_vowels:
        raise typer.BadParameter("--diphthongs and --all are mutually exclusive")
    mode = "diphthongs" if diphthongs else "all" if all_vowels else "mono"
    save_chart(session, mode=mode)


@app.command()
def run(
    session: str,
    gender: Annotated[Gender, typer.Option(help="Speaker gender (M/F/C)")] = Gender.M,
    diphthongs: Annotated[bool, typer.Option("--diphthongs", help="Show diphthongs only in plot", is_flag=True)] = False,
    all_vowels: Annotated[bool, typer.Option("--all", help="Show all vowels in plot", is_flag=True)] = False,
    min_sounding_interval: Annotated[float, typer.Option(help="Minimum sounding interval (s)")] = 0.12,
) -> None:
    """Run the full pipeline: silences → label → nucleus → formants → plot."""
    from vowels.pipeline.formants import extract_formants
    from vowels.pipeline.label import label_textgrid
    from vowels.pipeline.nucleus import make_nucleus_points
    from vowels.pipeline.silences import detect_silences
    from vowels.plots.vowel_space import save_chart

    if diphthongs and all_vowels:
        raise typer.BadParameter("--diphthongs and --all are mutually exclusive")
    mode = "diphthongs" if diphthongs else "all" if all_vowels else "mono"
    detect_silences(session, min_sounding_interval=min_sounding_interval)
    label_textgrid(session)
    make_nucleus_points(session, diphthongs=True)
    extract_formants(session, gender=gender.value)
    save_chart(session, mode=mode)
