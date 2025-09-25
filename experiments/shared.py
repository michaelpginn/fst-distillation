from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Literal, TypedDict

Task = Literal["inflection", "g2p", "transliteration"]


def find_data_file(task: Task, name: str):
    """Finds a file by name in the appropriate data folder"""
    root = Path(__file__).parent.parent / "data" / task
    matches = list(root.rglob(name))

    if len(matches) == 0:
        raise RuntimeError(f"Could not find file {name}!")
    if len(matches) == 1:
        return matches[0]

    # If we have multiple matches, check if it's a test file
    if name.endswith(".tst"):
        matches = [m for m in matches if "gold" in str(m).lower()]
        if len(matches) == 1:
            return matches[0]

    raise RuntimeError(f"Multiple possible matches: {matches}")


def add_task_parser(parser: ArgumentParser):
    parser.add_argument("task", choices=["inflection"])
    parser.add_argument(
        "dataset",
        required=True,
        help="The key for the dataset. Probably an isocode or other name.",
    )


class DataFiles(TypedDict):
    train: Path
    eval: Path
    test: Path
    train_aligned: Path
    eval_aligned: Path
    test_aligned: Path
    has_features: bool


def get_data_files(args: Namespace) -> DataFiles:
    return {
        "train": find_data_file(args.task, f"{args.dataset}.trn"),
        "eval": find_data_file(args.task, f"{args.dataset}.dev"),
        "test": find_data_file(args.task, f"{args.dataset}.tst"),
        "train_aligned": Path(__file__).parent
        / f"clustering/aligned_data/{args.language}.trn.aligned",
        "eval_aligned": Path(__file__).parent
        / f"clustering/aligned_data/{args.language}.dev.aligned",
        "test_aligned": Path(__file__).parent
        / f"clustering/aligned_data/{args.language}.tst.aligned",
        "has_features": args.task == "inflection",
    }
