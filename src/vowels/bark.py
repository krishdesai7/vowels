import polars as pl


def _z_expr(col: str, i: int) -> pl.Expr:
    c: pl.Expr = pl.col(col)
    return ((26.81 * c) / (1960 + c) - 0.53).alias(f"Z{i}")


def add_bark_dims(
    df: pl.DataFrame,
    *,
    f0_col: str = "F0",
    formant_cols: tuple[str, str, str] = ("F1", "F2", "F3"),
) -> pl.DataFrame:
    cols: tuple[str, ...] = (f0_col, *formant_cols)
    return df.with_columns(
        _z_expr(c, i) for i, c in enumerate(cols)
    ).with_columns(
        (pl.col("Z1") - pl.col("Z0")).alias("Openness"),
        (pl.col("Z2") - pl.col("Z1")).alias("Frontness"),
        (pl.col("Z3") - pl.col("Z2")).alias("Roundness"),
    )
