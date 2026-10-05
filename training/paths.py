"""Locate repository roots and source files for experiment snapshots."""
from pathlib import Path

def repository_root(file):
    for parent in Path(file).resolve().parents:
        if (parent / "AGENTS.md").exists() or (parent / "fixation_data").exists():
            return parent
    return Path(file).resolve().parents[1]

def source_path(root, name):
    root = Path(root)
    direct = root / name
    if direct.is_file():
        return direct
    matches = [root / folder / name for folder in
               ("training", "yin_tests", "kanwisher_tests", "data_preparation", "analysis", "reporting")
               if (root / folder / name).is_file()]
    if len(matches) != 1:
        raise FileNotFoundError(f"Expected one source for {name!r} under {root}; found {matches}")
    return matches[0]
