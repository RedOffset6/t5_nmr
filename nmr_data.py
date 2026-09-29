"""NMR/SMILES data loading and tokenization shared by training and evaluation."""

import json
import shutil
from contextlib import nullcontext
from pathlib import Path

from datasets import Dataset, DatasetDict, load_from_disk


DEFAULT_PREFIX = "predict SMILES from NMR spectrum: "

# Dataset split name -> file name part, e.g. src-val.txt for "validation".
SPLIT_FILES = {"train": "train", "validation": "val", "test": "test"}


def read_lines(path: Path) -> list[str]:
    if not path.is_file():
        raise FileNotFoundError(f"Required data file not found: {path}")
    with path.open("r", encoding="utf-8") as file:
        return [line.strip() for line in file]


def split_size(data_dir: Path, split: str) -> int:
    """Number of examples in a split, without loading the spectra."""
    with (data_dir / f"tgt-{SPLIT_FILES[split]}.txt").open("rb") as file:
        return sum(1 for _ in file)


def load_split(data_dir: Path, split: str) -> Dataset:
    """Load one split as a Dataset with "src" (spectrum) and "tgt" (SMILES)."""
    name = SPLIT_FILES[split]
    src_path = data_dir / f"src-{name}.txt"
    tgt_path = data_dir / f"tgt-{name}.txt"
    src_lines = read_lines(src_path)
    tgt_lines = read_lines(tgt_path)
    if len(src_lines) != len(tgt_lines):
        raise ValueError(
            "Line count mismatch:\n"
            f"{src_path}: {len(src_lines)} lines\n"
            f"{tgt_path}: {len(tgt_lines)} lines"
        )
    return Dataset.from_dict({"src": src_lines, "tgt": tgt_lines})


def build_generation_collate(tokenizer, prefix: str):
    """Collate raw examples into (encoded inputs, reference SMILES) for generate."""

    def collate(examples):
        encoded = tokenizer(
            [prefix + example["src"] for example in examples],
            truncation=False,
            padding=True,
            return_tensors="pt",
        )
        targets = [example["tgt"].strip() for example in examples]
        return dict(encoded), targets

    return collate


def tokenize_splits(
    data_dir: Path,
    tokenizer,
    prefix: str,
    target_max_length: int,
    num_proc: int,
) -> DatasetDict:
    """Tokenize the train and validation splits for training.

    The test split is only used for generation, which tokenizes raw text.
    Inputs are never truncated; targets are cut at target_max_length tokens.
    """

    def preprocess(examples):
        model_inputs = tokenizer(
            [prefix + source for source in examples["src"]],
            truncation=False,
        )
        labels = tokenizer(
            text_target=examples["tgt"],
            max_length=target_max_length,
            truncation=True,
        )
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    splits = DatasetDict(
        {split: load_split(data_dir, split) for split in ("train", "validation")}
    )
    return splits.map(
        preprocess,
        batched=True,
        remove_columns=["src", "tgt"],
        num_proc=max(num_proc, 1),
        desc="Tokenizing dataset",
    )


def load_tokenized_splits(
    data_dir: Path,
    tokenizer,
    model_name: str,
    prefix: str,
    target_max_length: int,
    cache_dir: Path,
    num_proc: int,
    is_main_process: bool = True,
    main_process_first=nullcontext,
) -> DatasetDict:
    """Tokenize once into an Arrow cache on disk, then memory-map it.

    Only the main process tokenizes; under torchrun the others wait inside
    main_process_first() and then load the finished cache. load_from_disk
    memory-maps the Arrow files, so the processes on a node share one copy of
    the data. A later job with the same settings skips tokenization.
    """
    fingerprint = {
        "model_name": model_name,
        "prefix": prefix,
        "target_max_length": target_max_length,
        "data_dir": str(data_dir.resolve()),
        "splits": ["train", "validation"],
    }
    fingerprint_path = cache_dir / "t5_train_fingerprint.json"

    with main_process_first():
        if is_main_process:
            if (
                fingerprint_path.is_file()
                and json.loads(fingerprint_path.read_text()) == fingerprint
            ):
                print("Using tokenized dataset cache:", cache_dir)
            else:
                # Write to a temporary folder and rename it at the end, so an
                # interrupted job never leaves a cache that looks complete.
                partial_dir = cache_dir.with_name(cache_dir.name + ".partial")
                shutil.rmtree(partial_dir, ignore_errors=True)
                tokenize_splits(
                    data_dir, tokenizer, prefix, target_max_length, num_proc
                ).save_to_disk(str(partial_dir))
                (partial_dir / fingerprint_path.name).write_text(
                    json.dumps(fingerprint, indent=4)
                )
                shutil.rmtree(cache_dir, ignore_errors=True)
                partial_dir.rename(cache_dir)
                print("Tokenized dataset saved to:", cache_dir)

        return load_from_disk(str(cache_dir))
