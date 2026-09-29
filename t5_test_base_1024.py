import os
import random
from pathlib import Path

import numpy as np
import torch
from datasets import Dataset, DatasetDict
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
MODEL_NAME = os.environ.get(
    "MODEL_NAME",
    "google/flan-t5-base",
)
DATA_DIR = Path(os.environ.get("DATA_DIR", BASE_DIR / "alberts_2d"))
OUTPUT_DIR = Path(
    os.environ.get(
        "OUTPUT_DIR",
        BASE_DIR / "outputs" / "flan-t5-base_nmr_input1024",
    )
)

SEED = 42
PREFIX = "predict SMILES from NMR spectrum: "
INPUT_MAX_LENGTH = 1024
TARGET_MAX_LENGTH = 128

LEARNING_RATE = 5e-5
TRAIN_BATCH_SIZE = 16
EVAL_BATCH_SIZE = 32
GRAD_ACCUMULATION_STEPS = 1
NUM_EPOCHS = 3
WEIGHT_DECAY = 0.01
SAVE_TOTAL_LIMIT = 2

# Generation is deliberately limited to a subset so evaluation cannot consume hours.
TEST_SAMPLE_SIZE = int(os.environ.get("TEST_SAMPLE_SIZE", "1000"))
GENERATION_BATCH_SIZE = int(os.environ.get("GENERATION_BATCH_SIZE", "32"))
GENERATION_MAX_NEW_TOKENS = 128


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


def print_configuration(dataset: DatasetDict, use_bf16: bool) -> None:
    print("===== Experiment Configuration =====")
    print("PyTorch:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))
    print("BF16 enabled:", use_bf16)
    print("Model:", MODEL_NAME)
    print("Data directory:", DATA_DIR)
    print("Output directory:", OUTPUT_DIR)
    print("Train rows:", len(dataset["train"]))
    print("Validation rows:", len(dataset["validation"]))
    print("Test rows:", len(dataset["test"]))
    print("Train batch size:", TRAIN_BATCH_SIZE)
    print("Gradient accumulation:", GRAD_ACCUMULATION_STEPS)
    print("Epochs:", NUM_EPOCHS)
    print("Generation test rows:", min(TEST_SAMPLE_SIZE, len(dataset["test"])))
    print("Input max length:", INPUT_MAX_LENGTH)
    print("Target max length:", TARGET_MAX_LENGTH)
    print("Eval batch size:", EVAL_BATCH_SIZE)
    print(
        "Effective train batch size:",
        TRAIN_BATCH_SIZE * GRAD_ACCUMULATION_STEPS,
    )

def report_length_statistics(
    dataset: Dataset,
    tokenizer,
    sample_size: int = 10000,
    batch_size: int = 256,
) -> None:
    """Report input token lengths before truncation."""

    sample_size = min(sample_size, len(dataset))

    # Sample with a fixed seed so every experiment measures the same examples.
    sample = (
        dataset.shuffle(seed=SEED)
        .select(range(sample_size))
    )

    lengths = []

    for start in range(0, sample_size, batch_size):
        end = min(start + batch_size, sample_size)
        sources = sample[start:end]["src"]

        encoded = tokenizer(
            [PREFIX + source for source in sources],
            truncation=False,
            padding=False,
            add_special_tokens=True,
        )

        lengths.extend(
            len(input_ids)
            for input_ids in encoded["input_ids"]
        )

    lengths = np.asarray(lengths)

    print("===== Input Length Statistics =====")
    print("Samples:", len(lengths))
    print("Mean:", float(np.mean(lengths)))
    print("Median:", float(np.median(lengths)))
    print("P90:", float(np.percentile(lengths, 90)))
    print("P95:", float(np.percentile(lengths, 95)))
    print("P99:", float(np.percentile(lengths, 99)))
    print("Maximum:", int(np.max(lengths)))

    for limit in (256, 512, 1024, 1536):
        print(
            f"Over {limit}:",
            f"{np.mean(lengths > limit):.2%}",
        )

    print(
        f"Over configured limit ({INPUT_MAX_LENGTH}):",
        f"{np.mean(lengths > INPUT_MAX_LENGTH):.2%}",
    )


def generate_test_metrics(model, tokenizer, test_dataset: Dataset) -> dict[str, float]:
    """Generate a bounded test subset and report exact string-match accuracy."""
    sample_count = min(TEST_SAMPLE_SIZE, len(test_dataset))
    if sample_count == 0:
        return {"test_exact_match": 0.0, "test_samples": 0}

    subset = test_dataset.select(range(sample_count))

    def collate(examples):
        inputs = [PREFIX + example["src"] for example in examples]
        encoded = tokenizer(
            inputs,
            max_length=INPUT_MAX_LENGTH,
            truncation=True,
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
    device = next(model.parameters()).device
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


def main() -> None:
    set_seed(SEED)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    dataset = load_splits(DATA_DIR)
    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    print_configuration(dataset, use_bf16)

    # Set LOCAL_FILES_ONLY=1 in an offline Slurm job after caching the model once.
    local_files_only = os.environ.get("LOCAL_FILES_ONLY", "0") == "1"
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        use_fast=True,
        local_files_only=local_files_only,
    )
    model = AutoModelForSeq2SeqLM.from_pretrained(
        MODEL_NAME,
        local_files_only=local_files_only,
    )

    report_length_statistics(
        dataset["train"],
        tokenizer,
    )

    def preprocess(examples):
        model_inputs = tokenizer(
            [PREFIX + source for source in examples["src"]],
            max_length=INPUT_MAX_LENGTH,
            truncation=True,
        )
        labels = tokenizer(
            text_target=examples["tgt"],
            max_length=TARGET_MAX_LENGTH,
            truncation=True,
        )
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    tokenized_dataset = dataset.map(
        preprocess,
        batched=True,
        remove_columns=["src", "tgt"],
        desc="Tokenizing dataset",
    )
    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

    training_args = Seq2SeqTrainingArguments(
        output_dir=str(OUTPUT_DIR),
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="steps",
        logging_steps=500,
        learning_rate=LEARNING_RATE,
        per_device_train_batch_size=TRAIN_BATCH_SIZE,
        per_device_eval_batch_size=EVAL_BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUMULATION_STEPS,
        gradient_checkpointing=False,
        weight_decay=WEIGHT_DECAY,
        save_total_limit=SAVE_TOTAL_LIMIT,
        num_train_epochs=NUM_EPOCHS,
        # Only calculate validation loss during training. Full sequence generation
        # previously took more than three hours after every epoch.
        predict_with_generate=False,
        bf16=use_bf16,
        fp16=torch.cuda.is_available() and not use_bf16,
        dataloader_pin_memory=torch.cuda.is_available(),
        push_to_hub=False,
        report_to="none",
        seed=SEED,
        data_seed=SEED,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["validation"],
        processing_class=tokenizer,
        data_collator=data_collator,
    )

    checkpoint = get_last_checkpoint(str(OUTPUT_DIR))
    if checkpoint:
        print("Resuming from checkpoint:", checkpoint)
    else:
        print("Starting training from the pretrained model")

    train_result = trainer.train(resume_from_checkpoint=checkpoint)
    trainer.save_metrics("train", train_result.metrics)
    trainer.save_state()

    final_model_dir = OUTPUT_DIR / "final_model"
    trainer.save_model(str(final_model_dir))
    tokenizer.save_pretrained(str(final_model_dir))
    print("Final model saved to:", final_model_dir)

    print("===== Bounded Generation Test =====")
    test_metrics = generate_test_metrics(model, tokenizer, dataset["test"])
    trainer.save_metrics("test", test_metrics)
    print(test_metrics)


if __name__ == "__main__":
    main()
