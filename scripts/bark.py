import os
from collections.abc import Callable
from pathlib import Path
from typing import Annotated

import polars as pl
import typer


def formant_midpoints(i: int) -> pl.Expr:
    col: pl.Expr = pl.col(f"F{i} Hz")
    return (
        col.str.replace_all(r"\s+", "")
        .str.split("-")
        .list.eval(pl.element().cast(pl.UInt16))
        .list.mean()
        .alias(f"F{i}")
    )


def bark_normalize(i: int) -> pl.Expr:
    col: pl.Expr = pl.col(f"F{i}")
    Z: pl.Expr = (26.81 * col) / (1960 + col) - 0.53
    return Z.alias(f"Z{i}")


def validate_input_file(input_file: Path) -> Path:
    suffix: str = input_file.suffix.lower()
    if suffix == ".csv":
        return input_file
    if suffix == ".parquet":
        if os.access(input_file, os.W_OK):
            return input_file
        raise typer.BadParameter("Input file is not writable.")
    raise typer.BadParameter("Input file must be a CSV or Parquet file.")


def resolve_output_file(output: Path | None, input_file: Path) -> Path:
    output = output or input_file.parent
    if output.is_dir():
        output /= input_file.with_suffix(".parquet").name
    if output.suffix.lower() == ".parquet":
        target: Path = output if output.exists() else output.parent
        if target.exists():
            if os.access(target, os.W_OK):
                return output
            raise typer.BadParameter(
                f"Output {'file' if output.exists() else 'directory'} is not writable."
            )
        raise typer.BadParameter("Parent directory of output does not exist.")
    raise typer.BadParameter("Output file must use the .parquet suffix.")


def main(
    input_file: Annotated[
        Path,
        typer.Argument(
            exists=True,
            dir_okay=False,
            resolve_path=True,
            callback=validate_input_file,
            help="Input file containing formant values or ranges (CSV or Parquet).",
        ),
    ],
    output: Annotated[
        Path | None,
        typer.Option(
            "--output",
            "-o",
            resolve_path=True,
            help="Output directory or parquet file to write the transformed data to.",
        ),
    ] = None,
) -> None:
    scan: Callable[[Path], pl.LazyFrame] = (
        pl.scan_parquet if input_file.suffix.lower() == ".parquet" else pl.scan_csv
    )
    output_file: Path = resolve_output_file(output, input_file)
    (
        scan(input_file)
        .with_columns(formant_midpoints(i) for i in range(1, 4))
        .with_columns(bark_normalize(i) for i in range(1, 4))
        .with_columns(
            Frontness=pl.col.Z2 - pl.col.Z1,
            Roundness=pl.col.Z3 - pl.col.Z2,
        )
        .select(
            "label",
            "category",
            "F1",
            "F2",
            "F3",
            "Frontness",
            "Roundness",
        )
        .sink_parquet(output_file)
    )


if __name__ == "__main__":
    typer.run(main)
