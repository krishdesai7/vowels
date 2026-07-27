import tkinter as tk
from pathlib import Path
from typing import Annotated

import polars as pl
import typer


def load_words(input_path: Path) -> pl.Series:
    frame: pl.DataFrame = pl.read_csv(input_path)

    if "word" not in frame.columns:
        raise ValueError(f"{input_path} has no 'word' column")

    words: pl.Series = frame.get_column("word")

    if words.is_empty():
        raise ValueError(f"{input_path} has no words")

    return words


def next_index(index: int, total: int) -> int:
    return min(index + 1, total - 1)


def previous_index(index: int) -> int:
    return max(index - 1, 0)


class ReadingListApp:
    def __init__(self, root: tk.Tk, words: pl.Series) -> None:
        self.root = root
        self.words = words
        self.index = 0

        root.title("Reading List")
        root.minsize(700, 400)

        self.word_label = tk.Label(root, font=("Cochineal", 72))
        self.word_label.pack(expand=True)

        self.progress_label = tk.Label(root, font=("Cochineal", 18))
        self.progress_label.pack(pady=(0, 20))

        buttons = tk.Frame(root)
        buttons.pack(pady=(0, 30))

        tk.Button(buttons, text="Previous", command=self.show_previous).pack(
            side="left", padx=10
        )
        tk.Button(buttons, text="Next", command=self.show_next).pack(
            side="left", padx=10
        )

        root.bind("<space>", self.show_next)
        root.bind("<Return>", self.show_next)
        root.bind("<Right>", self.show_next)
        root.bind("<Left>", self.show_previous)
        root.bind("<Escape>", lambda _event: root.destroy())

        self.render()

    def render(self) -> None:
        self.word_label.config(text=self.words[self.index])
        self.progress_label.config(text=f"{self.index + 1} / {len(self.words)}")

    def show_next(self, _event: tk.Event | None = None) -> None:
        self.index = next_index(self.index, self.words.len())
        self.render()

    def show_previous(self, _event: tk.Event | None = None) -> None:
        self.index = previous_index(self.index)
        self.render()


def main(
    input_path: Annotated[Path, typer.Argument()] = Path("data/labels.csv"),
) -> None:

    try:
        words: pl.Series = load_words(input_path)
    except (OSError, pl.exceptions.PolarsError, ValueError) as error:
        typer.echo(f"Error: {error}", err=True)
        raise typer.Exit(-1) from error
    else:
        root: tk.Tk = tk.Tk()
        ReadingListApp(root, words)
        root.mainloop()


if __name__ == "__main__":
    main()
