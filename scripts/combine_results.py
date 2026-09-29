"""Combine the chunk results of an evaluation array job into one score.

    python scripts/combine_results.py xl_4x4_10ep

Finds <output dir>/test_<start>_<end>_results.json for the run, checks that
the chunks cover the whole split exactly once, prints top-1 to top-N exact
match weighted by chunk size, joins the chunks' predictions into
prd-test.txt, and records the result in reports/evaluation_summary.tsv.
"""

import argparse
import csv
import datetime
import json
import re
import sys
from pathlib import Path

REPO_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_DIR))

from nmr_data import split_size  # noqa: E402


SUMMARY_COLUMNS = [
    "experiment",
    "chunks",
    "total_samples",
    "exact_match",
    "job_id",
    "date",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("run", help="Run name, as in configs/train/<run>.env")
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Folder with the chunk results; defaults to the run's OUTPUT_DIR",
    )
    parser.add_argument("--split", choices=("validation", "test"), default="test")
    parser.add_argument("--data-dir", type=Path, default=REPO_DIR / "alberts_2d")
    parser.add_argument(
        "--total",
        type=int,
        help="Examples in the split; read from --data-dir when not given",
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=REPO_DIR / "reports" / "evaluation_summary.tsv",
    )
    parser.add_argument(
        "--no-summary",
        action="store_true",
        help="Print the result without recording it in the summary",
    )
    return parser.parse_args()


def run_output_dir(run: str) -> Path:
    """OUTPUT_DIR from configs/train/<run>.env, or outputs/<run>."""
    config = REPO_DIR / "configs" / "train" / f"{run}.env"
    if not config.is_file():
        raise FileNotFoundError(f"No config {config}")
    for line in config.read_text().splitlines():
        if line.startswith("OUTPUT_DIR="):
            return REPO_DIR / line.split("=", 1)[1].strip().strip('"')
    return REPO_DIR / "outputs" / run


def find_chunks(output_dir: Path, split: str, total: int) -> list[dict]:
    pattern = re.compile(rf"{split}_(\d+)_(\d+)_results\.json")
    chunks = []
    for path in output_dir.iterdir():
        if pattern.fullmatch(path.name) or path.name == f"full_{split}_results.json":
            result = json.loads(path.read_text())
            result["path"] = path
            chunks.append(result)
    if not chunks:
        raise FileNotFoundError(f"No {split} chunk results in {output_dir}")

    chunks.sort(key=lambda chunk: chunk["start_index"])
    expected_start = 0
    for chunk in chunks:
        # Results from older versions of the evaluator have no status.
        if chunk.get("status", "complete") != "complete":
            raise ValueError(f"{chunk['path'].name} is not complete")
        if chunk["start_index"] != expected_start:
            raise ValueError(
                f"Chunks do not tile [0, {total}): expected a chunk starting at "
                f"{expected_start}, found {chunk['path'].name}. Remove results "
                "from earlier evaluations with different chunk sizes."
            )
        expected_start = chunk["end_index"]
    if expected_start != total:
        raise ValueError(
            f"Chunks cover [0, {expected_start}) but the split has {total} examples"
        )
    return chunks


def chunk_predictions_file(chunk: dict) -> Path:
    """The chunk's predictions file, next to its results."""
    # The results hold an absolute path, which breaks if the folder moved.
    name = Path(chunk.get("predictions_file", "")).name or (
        chunk["path"].name.removesuffix("_results.json") + "_predictions.txt"
    )
    return chunk["path"].with_name(name)


def combine_predictions(chunks: list[dict], destination: Path) -> None:
    with destination.open("w", encoding="utf-8") as output:
        for chunk in chunks:
            predictions = chunk_predictions_file(chunk)
            lines = predictions.read_text(encoding="utf-8").splitlines(keepends=True)
            expected = chunk["samples"] * chunk.get("num_outputs", 1)
            if len(lines) != expected:
                raise ValueError(
                    f"{predictions.name} has {len(lines)} lines, expected {expected}"
                )
            output.writelines(lines)


def record_summary(summary: Path, row: dict) -> None:
    rows = []
    if summary.is_file():
        with summary.open(newline="") as file:
            rows = list(csv.DictReader(file, delimiter="\t"))
    # Evaluating a run again replaces its row.
    rows = [existing for existing in rows if existing["experiment"] != row["experiment"]]
    rows.append(row)
    with summary.open("w", newline="") as file:
        writer = csv.DictWriter(
            file, fieldnames=SUMMARY_COLUMNS, delimiter="\t", restval="",
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir or run_output_dir(args.run)
    total = args.total or split_size(args.data_dir, args.split)
    chunks = find_chunks(output_dir, args.split, total)

    num_outputs = min(len(chunk.get("top_n_matches", [0])) for chunk in chunks)
    top_n = [
        sum(chunk.get("top_n_matches", [chunk["matches"]])[rank] for chunk in chunks)
        for rank in range(num_outputs)
    ]
    for rank, matches in enumerate(top_n, start=1):
        print(f"top-{rank}: {matches / total:.4f}")

    file_split = "val" if args.split == "validation" else "test"
    predictions = output_dir / f"prd-{file_split}.txt"
    if all(chunk_predictions_file(chunk).is_file() for chunk in chunks):
        combine_predictions(chunks, predictions)
        print("Predictions:", predictions)
    else:
        # Evaluations made before predictions were saved.
        print("Some chunks have no predictions file; prd file not written")

    if not args.no_summary:
        job_ids = sorted({str(chunk.get("slurm_job_id", "")) for chunk in chunks} - {""})
        record_summary(
            args.summary,
            {
                "experiment": args.run,
                "chunks": len(chunks),
                "total_samples": total,
                "exact_match": f"{top_n[0] / total:.8f}",
                "job_id": ",".join(job_ids),
                "date": datetime.date.today().isoformat(),
            },
        )
        print("Recorded in:", args.summary)


if __name__ == "__main__":
    main()
