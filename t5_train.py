import json
import os
import random
import shutil
from pathlib import Path

import numpy as np
import torch
from datasets import Dataset, DatasetDict, load_from_disk
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)
from transformers.trainer_utils import get_last_checkpoint


# Paths are resolved relative to this file, not the directory used to launch it.
BASE_DIR = Path(__file__).resolve().parent

# ===== Experiment identity =====

MODEL_NAME = os.environ.get(
    "MODEL_NAME",
    "google/flan-t5-base",
)

TARGET_MAX_LENGTH = int(
    os.environ.get("TARGET_MAX_LENGTH", "128")
)

DATA_DIR = Path(
    os.environ.get(
        "DATA_DIR",
        BASE_DIR / "alberts_2d",
    )
)

# Convert google/flan-t5-base into flan-t5-base.
MODEL_SHORT_NAME = MODEL_NAME.split("/")[-1]

# Each model receives an independent folder.
DEFAULT_OUTPUT_DIR = (
    BASE_DIR
    / "outputs"
    / f"{MODEL_SHORT_NAME}_nmr"
)

OUTPUT_DIR = Path(
    os.environ.get(
        "OUTPUT_DIR",
        DEFAULT_OUTPUT_DIR,
    )
)


TOKENIZED_CACHE_DIR = Path(
    os.environ.get(
        "TOKENIZED_CACHE_DIR",
        OUTPUT_DIR / "tokenized",
    )
)


# ===== Distributed training =====

# torchrun sets these for every process; a plain `python t5_train.py` run
# has none of them and trains in a single process.
WORLD_SIZE = int(os.environ.get("WORLD_SIZE", "1"))
RANK = int(os.environ.get("RANK", "0"))
LOCAL_RANK = int(os.environ.get("LOCAL_RANK", "0"))

# none: one process, one GPU. ddp: every GPU holds the full model.
# fsdp: parameters, gradients and optimizer state are sharded across GPUs.
PARALLEL_MODE = os.environ.get(
    "PARALLEL_MODE",
    "ddp" if WORLD_SIZE > 1 else "none",
).strip().lower()

if PARALLEL_MODE not in {"none", "ddp", "fsdp"}:
    raise ValueError(
        f"PARALLEL_MODE must be none, ddp or fsdp, got {PARALLEL_MODE!r}"
    )
if (PARALLEL_MODE == "none") != (WORLD_SIZE == 1):
    raise ValueError(
        f"PARALLEL_MODE={PARALLEL_MODE} does not match WORLD_SIZE={WORLD_SIZE}: "
        "use PARALLEL_MODE=none with plain python, ddp or fsdp with torchrun."
    )

DATALOADER_NUM_WORKERS = int(
    os.environ.get("DATALOADER_NUM_WORKERS", "4")
)

# 0 trains for NUM_EPOCHS; a positive value stops after that many optimizer
# steps, for short smoke tests of the whole pipeline.
MAX_STEPS = int(
    os.environ.get("MAX_STEPS", "0")
)


def is_main_process() -> bool:
    return RANK == 0


# ===== Reproducibility =====

SEED = int(os.environ.get("SEED", "42"))
PREFIX = os.environ.get(
    "PREFIX",
    "predict SMILES from NMR spectrum: ",
)


# ===== Optimisation =====

LEARNING_RATE = float(
    os.environ.get("LEARNING_RATE", "5e-5")
)

TRAIN_BATCH_SIZE = int(
    os.environ.get("TRAIN_BATCH_SIZE", "16")
)

EVAL_BATCH_SIZE = int(
    os.environ.get("EVAL_BATCH_SIZE", "32")
)

GRAD_ACCUMULATION_STEPS = int(
    os.environ.get("GRAD_ACCUMULATION_STEPS", "1")
)

NUM_EPOCHS = float(
    os.environ.get("NUM_EPOCHS", "3")
)

WEIGHT_DECAY = float(
    os.environ.get("WEIGHT_DECAY", "0.01")
)


# ===== Checkpointing =====

SAVE_STEPS = int(
    os.environ.get("SAVE_STEPS", "2000")
)

SAVE_TOTAL_LIMIT = int(
    os.environ.get("SAVE_TOTAL_LIMIT", "2")
)


# ===== Generation test =====

TEST_SAMPLE_SIZE = int(
    os.environ.get("TEST_SAMPLE_SIZE", "1000")
)

GENERATION_BATCH_SIZE = int(
    os.environ.get("GENERATION_BATCH_SIZE", "32")
)

GENERATION_MAX_NEW_TOKENS = int(
    os.environ.get("GENERATION_MAX_NEW_TOKENS", "128")
)

def get_env_bool(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)

    if value is None:
        return default

    return value.strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


GRADIENT_CHECKPOINTING = get_env_bool(
    "GRADIENT_CHECKPOINTING",
    False,
)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def read_lines(path: Path) -> list[str]:
    if not path.is_file():
        raise FileNotFoundError(f"Required data file not found: {path}")
    with path.open("r", encoding="utf-8") as file:
        return [line.strip() for line in file]


def build_split(src_path: Path, tgt_path: Path) -> Dataset:
    src_lines = read_lines(src_path)
    tgt_lines = read_lines(tgt_path)
    if len(src_lines) != len(tgt_lines):
        raise ValueError(
            "Line count mismatch:\n"
            f"{src_path}: {len(src_lines)} lines\n"
            f"{tgt_path}: {len(tgt_lines)} lines"
        )
    return Dataset.from_dict({"src": src_lines, "tgt": tgt_lines})


def load_splits(data_dir: Path) -> DatasetDict:
    return DatasetDict(
        {
            "train": build_split(
                data_dir / "src-train.txt", data_dir / "tgt-train.txt"
            ),
            "validation": build_split(
                data_dir / "src-val.txt", data_dir / "tgt-val.txt"
            ),
            "test": build_split(
                data_dir / "src-test.txt", data_dir / "tgt-test.txt"
            ),
        }
    )


def print_configuration(
    dataset: DatasetDict,
    use_bf16: bool,
) -> None:
    """Print the complete experiment configuration."""

    # One training process per GPU, so WORLD_SIZE is the number of GPUs.
    global_effective_batch_size = (
        TRAIN_BATCH_SIZE
        * GRAD_ACCUMULATION_STEPS
        * WORLD_SIZE
    )

    print("===== Experiment Configuration =====")

    print("PyTorch:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())
    print("Visible GPU count:", torch.cuda.device_count())

    if torch.cuda.is_available():
        print(
            f"GPU {LOCAL_RANK}:",
            torch.cuda.get_device_name(LOCAL_RANK),
        )

    print("Parallel mode:", PARALLEL_MODE)
    print("World size (GPUs):", WORLD_SIZE)
    print("Nodes:", os.environ.get("SLURM_JOB_NUM_NODES", "1"))

    print("BF16 enabled:", use_bf16)

    print("Model:", MODEL_NAME)
    print("Data directory:", DATA_DIR)
    print("Output directory:", OUTPUT_DIR)

    print("Train rows:", len(dataset["train"]))
    print("Validation rows:", len(dataset["validation"]))
    print("Test rows:", len(dataset["test"]))

    print("Target max length:", TARGET_MAX_LENGTH)

    print("Learning rate:", LEARNING_RATE)
    print("Weight decay:", WEIGHT_DECAY)
    print("Epochs:", NUM_EPOCHS)

    print(
        "Train batch size per device:",
        TRAIN_BATCH_SIZE,
    )
    print(
        "Gradient accumulation steps:",
        GRAD_ACCUMULATION_STEPS,
    )
    print(
        "Global effective train batch size:",
        global_effective_batch_size,
    )

    print("Evaluation batch size:", EVAL_BATCH_SIZE)
    print(
        "Generation batch size:",
        GENERATION_BATCH_SIZE,
    )
    print(
        "Generation test rows:",
        min(TEST_SAMPLE_SIZE, len(dataset["test"])),
    )
    print(
        "Generation max new tokens:",
        GENERATION_MAX_NEW_TOKENS,
    )

    print(
        "Gradient checkpointing:",
        GRADIENT_CHECKPOINTING,
    )
    print("Dataloader workers:", DATALOADER_NUM_WORKERS)
    print("Max steps:", MAX_STEPS if MAX_STEPS > 0 else "unlimited")
    print("Tokenized cache:", TOKENIZED_CACHE_DIR)

    print("Save strategy: steps")
    print("Save every optimizer steps:", SAVE_STEPS)
    print(
        "Checkpoint retention limit:",
        SAVE_TOTAL_LIMIT,
    )

    print("Random seed:", SEED)
    print("====================================")

def generate_test_metrics(
    model, tokenizer, test_dataset: Dataset, device: torch.device
) -> dict[str, float]:
    """Generate a bounded test subset and report exact string-match accuracy."""
    sample_count = min(TEST_SAMPLE_SIZE, len(test_dataset))
    if sample_count == 0:
        return {"test_exact_match": 0.0, "test_samples": 0}

    subset = test_dataset.select(range(sample_count))

    def collate(examples):
        inputs = [PREFIX + example["src"] for example in examples]
        encoded = tokenizer(
            inputs,
            truncation=False,
            padding=True,
            return_tensors="pt",
        )
        targets = [example["tgt"].strip() for example in examples]
        return encoded, targets

    loader = DataLoader(
        subset,
        batch_size=GENERATION_BATCH_SIZE,
        shuffle=False,
        collate_fn=collate,
    )
    model.eval()
    matches = 0

    with torch.inference_mode():
        for encoded, targets in loader:
            encoded = {key: value.to(device) for key, value in encoded.items()}
            generated = model.generate(
                **encoded,
                max_new_tokens=GENERATION_MAX_NEW_TOKENS,
                do_sample=False,
                num_beams=1,
            )
            predictions = tokenizer.batch_decode(
                generated, skip_special_tokens=True
            )
            matches += sum(
                prediction.strip() == target
                for prediction, target in zip(predictions, targets)
            )

    return {
        "test_exact_match": matches / sample_count,
        "test_samples": sample_count,
    }


def save_metrics_file(split: str, metrics: dict) -> None:
    """Write <split>_results.json and all_results.json like Trainer.save_metrics."""
    with (OUTPUT_DIR / f"{split}_results.json").open("w") as file:
        json.dump(metrics, file, indent=4, sort_keys=True)

    all_results_path = OUTPUT_DIR / "all_results.json"
    all_metrics = {}
    if all_results_path.is_file():
        with all_results_path.open() as file:
            all_metrics = json.load(file)
    all_metrics.update(metrics)
    with all_results_path.open("w") as file:
        json.dump(all_metrics, file, indent=4, sort_keys=True)


def tokenize_splits(tokenizer) -> DatasetDict:
    """Tokenise the train/validation/test splits read from DATA_DIR."""

    def preprocess(examples):
        model_inputs = tokenizer(
            [PREFIX + source for source in examples["src"]],
            truncation=False,
        )
        labels = tokenizer(
            text_target=examples["tgt"],
            max_length=TARGET_MAX_LENGTH,
            truncation=True,
        )
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    return load_splits(DATA_DIR).map(
        preprocess,
        batched=True,
        remove_columns=["src", "tgt"],
        num_proc=max(DATALOADER_NUM_WORKERS, 1),
        desc="Tokenizing dataset",
    )


def load_tokenized_dataset(tokenizer, training_args) -> DatasetDict:
    """Tokenise once per job on rank 0; every rank then memory-maps the result.

    load_from_disk memory-maps the Arrow files, so the processes on a node
    share one copy of the data instead of each holding its own.
    """
    fingerprint = {
        "model_name": MODEL_NAME,
        "prefix": PREFIX,
        "target_max_length": TARGET_MAX_LENGTH,
        "data_dir": str(DATA_DIR.resolve()),
    }
    fingerprint_path = TOKENIZED_CACHE_DIR / "t5_train_fingerprint.json"

    with training_args.main_process_first(local=False, desc="dataset tokenization"):
        if is_main_process():
            if (
                fingerprint_path.is_file()
                and json.loads(fingerprint_path.read_text()) == fingerprint
            ):
                print("Using tokenized dataset cache:", TOKENIZED_CACHE_DIR)
            else:
                # Write to a temporary folder and rename it at the end, so an
                # interrupted job never leaves a cache that looks complete.
                partial_dir = TOKENIZED_CACHE_DIR.with_name(
                    TOKENIZED_CACHE_DIR.name + ".partial"
                )
                shutil.rmtree(partial_dir, ignore_errors=True)
                tokenize_splits(tokenizer).save_to_disk(str(partial_dir))
                (partial_dir / fingerprint_path.name).write_text(
                    json.dumps(fingerprint, indent=4)
                )
                shutil.rmtree(TOKENIZED_CACHE_DIR, ignore_errors=True)
                partial_dir.rename(TOKENIZED_CACHE_DIR)
                print("Tokenized dataset saved to:", TOKENIZED_CACHE_DIR)

        return load_from_disk(str(TOKENIZED_CACHE_DIR))


def build_training_args(use_bf16: bool) -> Seq2SeqTrainingArguments:
    parallel_args = {}

    if PARALLEL_MODE == "ddp":
        # T5 uses every parameter in each step, so DDP can skip the search.
        parallel_args["ddp_find_unused_parameters"] = False

    if PARALLEL_MODE == "fsdp":
        # Keys as read by transformers 5.12 (TrainingArguments._process_fsdp_args).
        parallel_args["fsdp"] = True
        parallel_args["fsdp_config"] = {
            "version": 2,
            "reshard_after_forward": True,
            "auto_wrap_policy": "TRANSFORMER_BASED_WRAP",
            "transformer_layer_cls_to_wrap": ["T5Block"],
            # Only local rank 0 loads the pretrained weights into host RAM;
            # the other ranks receive them by broadcast.
            "cpu_ram_efficient_loading": True,
            # Checkpoints hold one shard per rank; the final model is
            # gathered into a single file in main().
            "state_dict_type": "SHARDED_STATE_DICT",
        }

    return Seq2SeqTrainingArguments(
        output_dir=str(OUTPUT_DIR),

        eval_strategy="epoch",

        save_strategy="steps",
        save_steps=SAVE_STEPS,
        save_total_limit=SAVE_TOTAL_LIMIT,

        logging_strategy="steps",
        logging_steps=500,

        learning_rate=LEARNING_RATE,
        per_device_train_batch_size=TRAIN_BATCH_SIZE,
        per_device_eval_batch_size=EVAL_BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUMULATION_STEPS,
        gradient_checkpointing=GRADIENT_CHECKPOINTING,

        weight_decay=WEIGHT_DECAY,
        num_train_epochs=NUM_EPOCHS,
        max_steps=MAX_STEPS if MAX_STEPS > 0 else -1,

        predict_with_generate=False,
        bf16=use_bf16,
        fp16=torch.cuda.is_available() and not use_bf16,

        dataloader_num_workers=DATALOADER_NUM_WORKERS,
        dataloader_pin_memory=torch.cuda.is_available(),
        push_to_hub=False,
        report_to="none",

        # Rank 0 tokenises the dataset and gathers the full XXL state dict
        # while the other ranks wait; both can exceed the 30-minute default.
        ddp_timeout=7200,

        seed=SEED,
        data_seed=SEED,

        **parallel_args,
    )


def main() -> None:
    set_seed(SEED)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()

    # Built first: it starts the distributed process group used below.
    training_args = build_training_args(use_bf16)

    # Set LOCAL_FILES_ONLY=1 in an offline Slurm job after caching the model once.
    local_files_only = os.environ.get("LOCAL_FILES_ONLY", "0") == "1"
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        use_fast=True,
        local_files_only=local_files_only,
    )

    tokenized_dataset = load_tokenized_dataset(tokenizer, training_args)
    if is_main_process():
        print_configuration(tokenized_dataset, use_bf16)

    if PARALLEL_MODE == "fsdp":
        # Lets from_pretrained see FSDP before the Trainer creates it, so
        # cpu_ram_efficient_loading applies to this load.
        os.environ["ACCELERATE_USE_FSDP"] = "true"
    model = AutoModelForSeq2SeqLM.from_pretrained(
        MODEL_NAME,
        local_files_only=local_files_only,
    )
    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["validation"],
        processing_class=tokenizer,
        data_collator=data_collator,
    )

    checkpoint = get_last_checkpoint(str(OUTPUT_DIR))
    if is_main_process():
        if checkpoint:
            print("Resuming from checkpoint:", checkpoint)
        else:
            print("Starting training from the pretrained model")

    train_result = trainer.train(resume_from_checkpoint=checkpoint)
    if is_main_process():
        trainer.save_metrics("train", train_result.metrics)
        trainer.save_state()

    final_model_dir = OUTPUT_DIR / "final_model"
    if trainer.is_fsdp_enabled:
        # Gather the shards into one Hugging Face model that loads anywhere.
        trainer.accelerator.state.fsdp_plugin.set_state_dict_type(
            "FULL_STATE_DICT"
        )
    # Called on every rank: under FSDP all ranks take part in the gather.
    trainer.save_model(str(final_model_dir))
    if is_main_process():
        tokenizer.save_pretrained(str(final_model_dir))
        print("Final model saved to:", final_model_dir)
        if torch.cuda.is_available():
            print(
                "Peak GPU memory allocated (GB):",
                round(torch.cuda.max_memory_allocated(LOCAL_RANK) / 1e9, 1),
            )

    test_dataset = None
    if is_main_process():
        test_dataset = build_split(
            DATA_DIR / "src-test.txt", DATA_DIR / "tgt-test.txt"
        )

    if PARALLEL_MODE == "none":
        print("===== Bounded Generation Test =====")
        device = next(model.parameters()).device
        test_metrics = generate_test_metrics(
            model, tokenizer, test_dataset, device
        )
        trainer.save_metrics("test", test_metrics)
        print(test_metrics)
        return

    # A DDP- or FSDP-wrapped model cannot generate on one rank, so rank 0
    # reloads the saved model in bf16, as evaluate_exact_match.py does.
    torch.distributed.barrier()
    del trainer, model
    torch.cuda.empty_cache()
    torch.distributed.destroy_process_group()
    if not is_main_process():
        return

    print("===== Bounded Generation Test =====")
    device = torch.device(
        f"cuda:{LOCAL_RANK}" if torch.cuda.is_available() else "cpu"
    )
    test_model = AutoModelForSeq2SeqLM.from_pretrained(
        str(final_model_dir),
        dtype=torch.bfloat16 if use_bf16 else torch.float32,
    ).to(device)
    test_metrics = generate_test_metrics(
        test_model, tokenizer, test_dataset, device
    )
    save_metrics_file("test", test_metrics)
    print(test_metrics)


if __name__ == "__main__":
    main()
