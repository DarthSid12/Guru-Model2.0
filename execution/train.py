"""Train a model with explicit curriculum and optimizer controls."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from execution.common.cli import launch

if __name__ == "__main__":
    launch("train", "model", __doc__)
