"""Fast tests of the evaluation bookkeeping; no model is loaded."""

import json
import sys
from pathlib import Path

import pytest

REPO_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_DIR))
sys.path.insert(0, str(REPO_DIR / "scripts"))

from combine_results import find_chunks, record_summary  # noqa: E402
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


def test_find_chunks_accepts_exact_tiling(tmp_path):
    write_chunk(tmp_path, 5, 10, [1, 2])
    write_chunk(tmp_path, 0, 5, [3, 4])
    chunks = find_chunks(tmp_path, "test", 10)
    assert [chunk["start_index"] for chunk in chunks] == [0, 5]


@pytest.mark.parametrize(
    "ranges",
    [
        [(0, 5)],  # missing the end
        [(0, 5), (4, 10)],  # overlap
        [(0, 4), (5, 10)],  # gap
    ],
)
def test_find_chunks_rejects_bad_tiling(tmp_path, ranges):
    for start, end in ranges:
        write_chunk(tmp_path, start, end, [0])
    with pytest.raises(ValueError):
        find_chunks(tmp_path, "test", 10)


def test_record_summary_replaces_row_of_same_run(tmp_path):
    summary = tmp_path / "summary.tsv"
    row = {"experiment": "xl", "chunks": 4, "total_samples": 10, "exact_match": "0.5"}
    record_summary(summary, row)
    record_summary(summary, {**row, "exact_match": "0.6"})
    lines = summary.read_text().splitlines()
    assert len(lines) == 2
    assert lines[1].split("\t")[3] == "0.6"
