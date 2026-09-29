"""Throughput and scaling efficiency of short runs on different node counts.

    for n in 1 2 4 8; do MAX_STEPS=100 ./submit.sh train xl_pg4_10ep --nodes $n; done
    python scripts/scaling_table.py 'outputs/xl_pg4_10ep_gb*_max100steps'

For each output folder, reads run_config.json and train_results.json and
prints GPUs, global batch, samples per second, and efficiency: samples per
second per GPU relative to the run with the fewest GPUs. Add nodes while the
efficiency stays above about 80%.
"""

import argparse
import glob
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "pattern", help="Glob of output folders, quoted so the shell doesn't expand it"
    )
    parser.add_argument(
        "--project-examples",
        type=int,
        default=0,
        help="Also print the hours to train this many examples, e.g. 6792130 for 10 epochs",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = []
    for folder in sorted(glob.glob(args.pattern)):
        folder = Path(folder)
        config_path = folder / "run_config.json"
        results_path = folder / "train_results.json"
        if not (config_path.is_file() and results_path.is_file()):
            print(f"Skipping {folder}: no run_config.json or train_results.json yet")
            continue
        config = json.loads(config_path.read_text())
        results = json.loads(results_path.read_text())
        rows.append(
            {
                "folder": folder.name,
                "gpus": config["world_size"],
                "global_batch": config["global_batch_size"],
                "samples_per_second": results["train_samples_per_second"],
            }
        )
    if not rows:
        raise SystemExit(f"No finished runs match {args.pattern}")

    rows.sort(key=lambda row: row["gpus"])
    reference = rows[0]["samples_per_second"] / rows[0]["gpus"]
    header = "| GPUs | Global batch | Samples/s | Samples/s per GPU | Efficiency |"
    if args.project_examples:
        header += f" Hours for {args.project_examples:,} examples |"
    print(header)
    print("|" + "---|" * (header.count("|") - 1))
    for row in rows:
        per_gpu = row["samples_per_second"] / row["gpus"]
        line = (
            f"| {row['gpus']} | {row['global_batch']} | {row['samples_per_second']:.1f} "
            f"| {per_gpu:.2f} | {per_gpu / reference:.0%} |"
        )
        if args.project_examples:
            line += f" {args.project_examples / row['samples_per_second'] / 3600:.1f} |"
        print(line)


if __name__ == "__main__":
    main()
