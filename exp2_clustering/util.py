from pathlib import Path


def find_data_file(name: str):
    """Finds a file by name in the task0-data folder"""
    root = Path(__file__).parent.parent / "task0-data"
    if name.endswith(".tst"):
        root /= "GOLD-TEST"

    for path in root.rglob(name):
        return path
    raise ValueError(f"Could not find file {name}!")
