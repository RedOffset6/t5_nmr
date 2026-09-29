"""Write a tiny synthetic dataset with the same layout as alberts_2d/.

    python tests/make_tiny_dataset.py [output_dir]

Each split has 64 rows by default. The spectra and SMILES are made up; the
dataset only exercises the pipeline, it teaches the model nothing.
"""

import sys
from pathlib import Path

REPO_DIR = Path(__file__).resolve().parent.parent

SMILES = ["CCO", "CC(=O)O", "c1ccccc1", "CCN(CC)CC", "C/C=C/C", "CC(C)O", "OC1CCCCC1", "C#N"]


def spectrum(row: int) -> str:
    shift = 0.5 + (row * 37 % 900) / 100
    return f"1HNMR {shift:.2f} {1 + row % 3}H m {7.26 - row % 5 / 10:.2f} 1H s | 13CNMR {20 + row % 150}.0"


def make_tiny_dataset(output_dir: Path, rows: int = 64) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    for split in ("train", "val", "test"):
        offset = {"train": 0, "val": 1000, "test": 2000}[split]
        (output_dir / f"src-{split}.txt").write_text(
            "".join(spectrum(offset + row) + "\n" for row in range(rows))
        )
        (output_dir / f"tgt-{split}.txt").write_text(
            "".join(SMILES[(offset + row) % len(SMILES)] + "\n" for row in range(rows))
        )
    return output_dir


if __name__ == "__main__":
    target = Path(sys.argv[1]) if len(sys.argv) > 1 else REPO_DIR / "tmp" / "tiny"
    print(make_tiny_dataset(target))
