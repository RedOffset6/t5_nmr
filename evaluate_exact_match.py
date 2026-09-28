"""Exact-match evaluation for an already trained FLAN-T5 model."""

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import torch
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_PREFIX = "predict SMILES from NMR spectrum: "


class TextPairDataset(Dataset):
    """A minimal dataset containing aligned source and target strings."""

    def __init__(self, sources: Sequence[str], targets: Sequence[str]) -> None:
        if len(sources) != len(targets):
            raise ValueError(
                f"Source/target length mismatch: {len(sources)} != {len(targets)}"
            )
        self.sources = list(sources)
        self.targets = list(targets)

    def __len__(self) -> int:
        return len(self.sources)

    def __getitem__(self, index: int) -> Tuple[str, str]:
        return self.sources[index], self.targets[index]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate exact string match without training the model."
    )
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=BASE_DIR / "alberts_2d",
    )
    parser.add_argument(
        "--split",
        choices=("validation", "test"),
        default="test",
    )
    parser.add_argument("--input-max-length", type=int, required=True)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument(
        "--start-index",
        type=int,
        default=0,
        help="First example to evaluate, using zero-based indexing.",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=0,
        help="Examples to evaluate; 0 means all examples after start-index.",
    )
    parser.add_argument("--prefix", default=DEFAULT_PREFIX)
    parser.add_argument("--output-file", type=Path, default=None)
    parser.add_argument(
        "--save-every",
        type=int,
        default=100,
        help="Save partial counts every N batches; 0 disables partial saves.",
    )
    parser.add_argument("--local-files-only", action="store_true")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if not args.model_path.is_dir():
        raise FileNotFoundError(f"Model directory not found: {args.model_path}")
    if not args.data_dir.is_dir():
        raise FileNotFoundError(f"Data directory not found: {args.data_dir}")
    if args.input_max_length <= 0:
        raise ValueError("--input-max-length must be positive")
    if args.max_new_tokens <= 0:
        raise ValueError("--max-new-tokens must be positive")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if args.start_index < 0:
        raise ValueError("--start-index must be non-negative")
    if args.sample_size < 0:
        raise ValueError("--sample-size must be non-negative")
    if args.save_every < 0:
        raise ValueError("--save-every must be non-negative")


def read_lines(path: Path) -> List[str]:
    if not path.is_file():
        raise FileNotFoundError(f"Required data file not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        return [line.strip() for line in handle]


def load_split(data_dir: Path, split: str) -> TextPairDataset:
    file_split = "val" if split == "validation" else "test"
    sources = read_lines(data_dir / f"src-{file_split}.txt")
    targets = read_lines(data_dir / f"tgt-{file_split}.txt")
    return TextPairDataset(sources, targets)


def choose_range(
    total_count: int,
    start_index: int,
    sample_size: int,
) -> Tuple[int, int]:
    if total_count == 0:
        raise ValueError("The selected split is empty")
    if start_index >= total_count:
        raise ValueError(
            f"--start-index {start_index} is outside a split containing "
            f"{total_count} examples"
        )
    if sample_size == 0:
        end_index = total_count
    else:
        end_index = min(start_index + sample_size, total_count)
    return start_index, end_index


def default_output_file(
    model_path: Path,
    split: str,
    start_index: int,
    end_index: int,
    total_count: int,
) -> Path:
    output_dir = (
        model_path.parent if model_path.name == "final_model" else model_path
    )
    if start_index == 0 and end_index == total_count:
        filename = f"full_{split}_results.json"
    else:
        filename = f"{split}_{start_index}_{end_index}_results.json"
    return output_dir / filename


def write_json(path: Path, data: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, ensure_ascii=False)
        handle.write("\n")


def main() -> None:
    args = parse_args()
    validate_args(args)

    full_dataset = load_split(args.data_dir, args.split)
    total_count = len(full_dataset)
    start_index, end_index = choose_range(
        total_count,
        args.start_index,
        args.sample_size,
    )
    selected_dataset = torch.utils.data.Subset(
        full_dataset,
        range(start_index, end_index),
    )
    selected_count = end_index - start_index

    output_file = args.output_file or default_output_file(
        args.model_path,
        args.split,
        start_index,
        end_index,
        total_count,
    )
    partial_file = output_file.with_suffix(".partial.json")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda" and torch.cuda.is_bf16_supported():
        model_dtype = torch.bfloat16
    elif device.type == "cuda":
        model_dtype = torch.float16
    else:
        model_dtype = torch.float32

    print("===== Exact-match evaluation =====", flush=True)
    print("Model:", args.model_path, flush=True)
    print("Split:", args.split, flush=True)
    print(f"Range: [{start_index}, {end_index})", flush=True)
    print(f"Samples: {selected_count} / {total_count}", flush=True)
    print("Input max length:", args.input_max_length, flush=True)
    print("Batch size:", args.batch_size, flush=True)
    print("Device:", device, flush=True)
    print("Dtype:", model_dtype, flush=True)
    print("Output:", output_file, flush=True)

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        use_fast=True,
        local_files_only=args.local_files_only,
    )
    model = AutoModelForSeq2SeqLM.from_pretrained(
        args.model_path,
        dtype=model_dtype,
        local_files_only=args.local_files_only,
    )
    model.to(device)
    model.eval()

    def collate_batch(
        examples: Sequence[Tuple[str, str]],
    ) -> Tuple[Dict[str, torch.Tensor], List[str]]:
        sources = [source for source, _ in examples]
        targets = [target.strip() for _, target in examples]
        encoded = tokenizer(
            [args.prefix + source for source in sources],
            max_length=args.input_max_length,
            truncation=True,
            padding=True,
            return_tensors="pt",
        )
        return dict(encoded), targets

    loader = DataLoader(
        selected_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_batch,
    )

    matches = 0
    processed = 0
    started_at = time.monotonic()

    with torch.inference_mode():
        progress = tqdm(loader, desc="Generating", unit="batch")
        for batch_number, (encoded, targets) in enumerate(progress, start=1):
            encoded = {
                name: tensor.to(device)
                for name, tensor in encoded.items()
            }
            generated = model.generate(
                **encoded,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                num_beams=1,
            )
            predictions = tokenizer.batch_decode(
                generated,
                skip_special_tokens=True,
            )
            matches += sum(
                prediction.strip() == target
                for prediction, target in zip(predictions, targets)
            )
            processed += len(targets)

            if args.save_every and batch_number % args.save_every == 0:
                write_json(
                    partial_file,
                    {
                        "status": "in_progress",
                        "split": args.split,
                        "start_index": start_index,
                        "end_index": start_index + processed,
                        "requested_end_index": end_index,
                        "samples": processed,
                        "matches": matches,
                        "exact_match": matches / processed,
                        "elapsed_seconds": time.monotonic() - started_at,
                    },
                )

    elapsed_seconds = time.monotonic() - started_at
    results: Dict[str, object] = {
        "status": "complete",
        "model_path": str(args.model_path.resolve()),
        "data_dir": str(args.data_dir.resolve()),
        "split": args.split,
        "start_index": start_index,
        "end_index": end_index,
        "samples": processed,
        "matches": matches,
        "exact_match": matches / processed,
        "input_max_length": args.input_max_length,
        "max_new_tokens": args.max_new_tokens,
        "batch_size": args.batch_size,
        "elapsed_seconds": elapsed_seconds,
    }
    write_json(output_file, results)
    write_json(partial_file, results)

    print("===== Results =====", flush=True)
    print(json.dumps(results, indent=2, ensure_ascii=False), flush=True)
    print("Saved to:", output_file, flush=True)


if __name__ == "__main__":
    main()
