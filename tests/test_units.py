"""Fast tests of the evaluation bookkeeping; no model is loaded."""

import json
import sys
from pathlib import Path

import pytest

REPO_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_DIR))
sys.path.insert(0, str(REPO_DIR / "scripts"))

from combine_results import check_tiling, load_chunks, record_summary  # noqa: E402
from evaluate_exact_match import count_match, resume_predictions  # noqa: E402


def test_count_match_counts_first_rank_and_below():
    matches = [0, 0, 0]
    count_match(matches, ["CCN", "CCO", "C"], "CCO")
    assert matches == [0, 1, 1]
    count_match(matches, ["C", "N", "O"], "CCO")
    assert matches == [0, 1, 1]


def test_resume_cuts_back_to_last_whole_spectrum(tmp_path):
    predictions = tmp_path / "predictions.txt"
    # Two outputs per spectrum: spectrum 0 and 1 complete, spectrum 2 has
    # one line and half of another.
    predictions.write_text("CCO\nCC\nN\nCCN\nC#N\nCC(")
    done, matches = resume_predictions(predictions, ["CCO", "CCN", "C#N"], 2)
    assert done == 2
    assert matches == [1, 2]
    assert predictions.read_text() == "CCO\nCC\nN\nCCN\n"


def test_resume_without_file_starts_at_zero(tmp_path):
    assert resume_predictions(tmp_path / "missing.txt", ["C"], 3) == (0, [0, 0, 0])


def write_chunk(directory: Path, start: int, end: int, matches: list[int]) -> None:
    (directory / f"test_{start}_{end}_results.json").write_text(
        json.dumps(
            {
                "status": "complete",
                "start_index": start,
                "end_index": end,
                "samples": end - start,
                "matches": matches[0],
                "top_n_matches": matches,
            }
        )
    )


def test_chunks_accepted_when_they_tile_the_split(tmp_path):
    write_chunk(tmp_path, 5, 10, [1, 2])
    write_chunk(tmp_path, 0, 5, [3, 4])
    chunks = load_chunks(tmp_path, "test")
    assert [chunk["start_index"] for chunk in chunks] == [0, 5]
    check_tiling(chunks, 10)


@pytest.mark.parametrize(
    "ranges",
    [
        [(0, 5)],  # missing the end
        [(0, 5), (4, 10)],  # overlap
        [(0, 4), (5, 10)],  # gap
    ],
)
def test_chunks_rejected_when_they_dont_tile_the_split(tmp_path, ranges):
    for start, end in ranges:
        write_chunk(tmp_path, start, end, [0])
    with pytest.raises(ValueError):
        check_tiling(load_chunks(tmp_path, "test"), 10)


def test_record_summary_replaces_row_of_same_run(tmp_path):
    summary = tmp_path / "summary.tsv"
    row = {"experiment": "xl", "chunks": 4, "total_samples": 10, "exact_match": "0.5"}
    record_summary(summary, row)
    record_summary(summary, {**row, "exact_match": "0.6"})
    lines = summary.read_text().splitlines()
    assert len(lines) == 2
    assert lines[1].split("\t")[3] == "0.6"


def test_split_size_read_from_results(tmp_path):
    import argparse

    from combine_results import split_total

    write_chunk(tmp_path, 0, 10, [1])
    chunks = load_chunks(tmp_path, "test")
    for chunk in chunks:
        chunk["split_size"] = 10
    args = argparse.Namespace(total=None, data_dir=None, run="unused", split="test")
    assert split_total(args, chunks) == 10


def make_config(monkeypatch, **env):
    from config import TrainConfig

    for name in ("WORLD_SIZE", "PARALLEL_MODE", "TRAIN_BATCH_SIZE", "PER_GPU_BATCH"):
        monkeypatch.delenv(name, raising=False)
    for name, value in env.items():
        monkeypatch.setenv(name, str(value))
    return TrainConfig.from_env()


@pytest.mark.parametrize(
    "scaling, expected",
    [("none", 5e-5), ("sqrt", 5e-5 * 4), ("linear", 5e-5 * 16)],
)
def test_learning_rate_scales_with_global_batch(monkeypatch, scaling, expected):
    # 4 per GPU x 64 GPUs = global batch 256, 16x the base batch.
    config = make_config(
        monkeypatch, PER_GPU_BATCH=4, WORLD_SIZE=64, PARALLEL_MODE="ddp", LR_SCALING=scaling
    )
    assert config.global_batch_size == 256
    assert config.effective_learning_rate == pytest.approx(expected)


def test_resume_refused_with_different_gpu_count(monkeypatch, tmp_path):
    from t5_train import check_resume_compatible

    written = make_config(
        monkeypatch, OUTPUT_DIR=tmp_path, PER_GPU_BATCH=4, WORLD_SIZE=8, PARALLEL_MODE="ddp"
    )
    written.write_json(tmp_path / "run_config.json")

    same = make_config(
        monkeypatch, OUTPUT_DIR=tmp_path, PER_GPU_BATCH=4, WORLD_SIZE=8, PARALLEL_MODE="ddp"
    )
    check_resume_compatible(same, str(tmp_path / "checkpoint-10"))

    fewer = make_config(
        monkeypatch, OUTPUT_DIR=tmp_path, PER_GPU_BATCH=4, WORLD_SIZE=4, PARALLEL_MODE="ddp"
    )
    with pytest.raises(SystemExit):
        check_resume_compatible(fewer, str(tmp_path / "checkpoint-10"))
