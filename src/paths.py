"""Core class for managing various paths."""

from argparse import ArgumentParser, Namespace
from os import PathLike
from pathlib import Path
from typing import TypedDict

from src.data.aligned.example import ALIGNMENT_SYMBOL


class Paths(TypedDict):
    identifier: str

    train: Path
    eval: Path
    test: Path
    has_features: bool

    train_aligned: Path
    eval_aligned: Path
    test_aligned: Path
    aligned_folder: Path
    alignment_symbol: str
    full_domain_aligned: Path

    models_folder: Path


def create_arg_parser():
    parser = ArgumentParser()
    parser.add_argument("data_folder", help="Folder containing raw files")
    parser.add_argument(
        "dataset", help="Name of raw data files, preceding .trn, .dev, and .tst"
    )
    parser.add_argument(
        "--features",
        action="store_true",
        help="Set this flag if the data includes a features column",
    )
    parser.add_argument("--models", help="A folder to store models in. Optional.")
    parser.add_argument("--alignment-symbol", default=ALIGNMENT_SYMBOL)
    return parser


def create_paths_from_args(args: Namespace):
    return create_paths(
        data_folder=args.data_folder,
        dataset=args.dataset,
        has_features=args.features or False,
        models_folder=args.models,
        alignment_symbol=args.alignment_symbol,
    )


def create_paths(
    data_folder: str | PathLike,
    dataset: str,
    has_features: bool,
    models_folder: str | PathLike | None,
    alignment_symbol: str = ALIGNMENT_SYMBOL,
) -> Paths:
    """Creates a dict with paths for all of the necessary files.

    Args:
        data_folder: A path to a folder containing the raw data files
        dataset: The name of the raw data files (followed by .trn, .dev, .tst)
        has_features: Whether the raw data includes a features column
        models_folder: A path to store models. If not provided, uses `<data_folder>/models`
    """
    data_root = Path(data_folder)
    assert data_root.exists()

    if models_folder:
        models_folder = Path(models_folder)
    else:
        models_folder = Path("./models")
    models_folder.mkdir(exist_ok=True)

    return {
        "identifier": data_root.stem + "." + dataset,
        "train": find_data_file(f"{dataset}.trn", data_root),
        "eval": find_data_file(f"{dataset}.dev", data_root),
        "test": find_data_file(f"{dataset}.tst", data_root),
        "has_features": has_features,
        "train_aligned": data_root / "aligned" / f"{dataset}.trn.aligned",
        "eval_aligned": data_root / "aligned" / f"{dataset}.dev.aligned",
        "test_aligned": data_root / "aligned" / f"{dataset}.tst.aligned",
        "aligned_folder": data_root / "aligned",
        "alignment_symbol": alignment_symbol,
        "full_domain_aligned": data_root / "aligned" / f"{dataset}.full.aligned",
        "models_folder": models_folder,
    }


def find_data_file(name: str, root_folder: Path):
    """Finds a file by name in the appropriate data folder"""
    matches = list(root_folder.rglob(name))

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
