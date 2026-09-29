import nltk
import evaluate
import numpy as np
import torch
from datasets import load_dataset
from transformers import T5Tokenizer, DataCollatorForSeq2Seq
from transformers import T5ForConditionalGeneration, Seq2SeqTrainingArguments, Seq2SeqTrainer


print("===== Device Check =====")
print("torch version:", torch.__version__)
print("torch cuda:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
print("device count:", torch.cuda.device_count())

if torch.cuda.is_available():
    print("device name:", torch.cuda.get_device_name(0))
    device = torch.device("cuda")
else:
    print("Using CPU")
    device = torch.device("cpu")
    

# =========================
# Global Config
# =========================

# Model
MODEL_NAME = "google/flan-t5-xl"

# Dataset
DATASET_NAME = "sentence-transformers/yahoo-answers"
DATASET_CONFIG = "question-answer-pair"
DATA_SPLIT = "train[:10000]"
TEST_SIZE = 0.2
SEED = 42

# Tokenization
INPUT_MAX_LENGTH = 128
TARGET_MAX_LENGTH = 64

# Training
OUTPUT_DIR = "./outputs/flan-t5-xl_yahoo10000_ep3_bs2_acc8_lr3e-5"
L_RATE = 3e-5
BATCH_SIZE = 2
PER_DEVICE_EVAL_BATCH = 2
GRAD_ACCUM_STEPS = 8
WEIGHT_DECAY = 0.01
SAVE_TOTAL_LIM = 1
NUM_EPOCHS = 3
GENERATION_MAX_LENGTH = 64

# =========================
# Print Experiment Config
# =========================

print("===== Experiment Config =====")
print("MODEL_NAME:", MODEL_NAME)
print("DATASET_NAME:", DATASET_NAME)
print("DATASET_CONFIG:", DATASET_CONFIG)
print("DATA_SPLIT:", DATA_SPLIT)
print("TEST_SIZE:", TEST_SIZE)
print("SEED:", SEED)

print("INPUT_MAX_LENGTH:", INPUT_MAX_LENGTH)
print("TARGET_MAX_LENGTH:", TARGET_MAX_LENGTH)

print("OUTPUT_DIR:", OUTPUT_DIR)
print("L_RATE:", L_RATE)
print("BATCH_SIZE:", BATCH_SIZE)
print("PER_DEVICE_EVAL_BATCH:", PER_DEVICE_EVAL_BATCH)
print("GRAD_ACCUM_STEPS:", GRAD_ACCUM_STEPS)
print("EFFECTIVE_BATCH_SIZE:", BATCH_SIZE * GRAD_ACCUM_STEPS)
print("WEIGHT_DECAY:", WEIGHT_DECAY)
print("SAVE_TOTAL_LIM:", SAVE_TOTAL_LIM)
print("NUM_EPOCHS:", NUM_EPOCHS)
print("GENERATION_MAX_LENGTH:", GENERATION_MAX_LENGTH)

print("bf16 supported:", torch.cuda.is_available() and torch.cuda.is_bf16_supported())

# Load the tokenizer, model, and data collator
tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME)
model = T5ForConditionalGeneration.from_pretrained(MODEL_NAME)
data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)


dataset = load_dataset(
    DATASET_NAME,
    DATASET_CONFIG,
    split=DATA_SPLIT
)

yahoo_answers_qa = dataset.train_test_split(
    test_size=TEST_SIZE,
    seed=SEED
)

print("===== Dataset Split =====")
print(yahoo_answers_qa)
print("Train size:", len(yahoo_answers_qa["train"]))
print("Test size:", len(yahoo_answers_qa["test"]))
print("Example train sample:", yahoo_answers_qa["train"][0])



# We prefix our tasks with "answer the question"
prefix = "Please answer this question: "

# Define the preprocessing function

def preprocess_function(examples):
   """Add prefix to the sentences, tokenize the text, and set the labels"""
   # The "inputs" are the tokenized answer:
   inputs = [prefix + doc for doc in examples["question"]]
   model_inputs = tokenizer(
      inputs,
      max_length=INPUT_MAX_LENGTH,
      truncation=True
   )
  
   # The "labels" are the tokenized outputs:
   labels = tokenizer(
      text_target=examples["answer"],
      max_length=TARGET_MAX_LENGTH,
      truncation=True
   )

   model_inputs["labels"] = labels["input_ids"]
   return model_inputs

# Map the preprocessing function across our dataset
tokenized_dataset = yahoo_answers_qa.map(preprocess_function, batched=True)

print("===== Tokenized Dataset =====")
print(tokenized_dataset)
print("Tokenized train size:", len(tokenized_dataset["train"]))
print("Tokenized test size:", len(tokenized_dataset["test"]))
print("Example tokenized sample:", tokenized_dataset["train"][0].keys())



nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)

metric = evaluate.load("rouge")



def compute_metrics(eval_preds):
    preds, labels = eval_preds

    # Some Trainer versions wrap predictions in a tuple.
    if isinstance(preds, tuple):
        preds = preds[0]

    preds = np.asarray(preds)
    labels = np.asarray(labels)

    # Defensive handling:
    # if logits are returned, convert them to token IDs.
    # logits shape: (batch, sequence_length, vocab_size)
    if preds.ndim == 3:
        preds = np.argmax(preds, axis=-1)

    pad_token_id = tokenizer.pad_token_id

    if pad_token_id is None:
        pad_token_id = 0

    # Cast to an integer type the tokenizer can safely handle.
    preds = preds.astype(np.int64)
    labels = labels.astype(np.int64)

    # Replace invalid token IDs in the predictions.
    preds = np.where(
        (preds >= 0) & (preds < tokenizer.vocab_size),
        preds,
        pad_token_id
    )

    # -100 is the ignore index used by the loss and cannot be decoded by the tokenizer.
    labels = np.where(
        labels != -100,
        labels,
        pad_token_id
    )

    decoded_preds = tokenizer.batch_decode(
        preds,
        skip_special_tokens=True
    )

    decoded_labels = tokenizer.batch_decode(
        labels,
        skip_special_tokens=True
    )

    decoded_preds = [
        "\n".join(nltk.sent_tokenize(pred.strip()))
        for pred in decoded_preds
    ]

    decoded_labels = [
        "\n".join(nltk.sent_tokenize(label.strip()))
        for label in decoded_labels
    ]

    result = metric.compute(
        predictions=decoded_preds,
        references=decoded_labels,
        use_stemmer=True
    )

    return result


# Global Parameters

training_args = Seq2SeqTrainingArguments(
    output_dir=OUTPUT_DIR,
    eval_strategy="epoch",
    logging_strategy="epoch",

    learning_rate=L_RATE,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=PER_DEVICE_EVAL_BATCH,
    gradient_accumulation_steps=GRAD_ACCUM_STEPS,

    gradient_checkpointing=True,

    weight_decay=WEIGHT_DECAY,
    save_total_limit=SAVE_TOTAL_LIM,
    num_train_epochs=NUM_EPOCHS,

    predict_with_generate=True,
    generation_max_length=GENERATION_MAX_LENGTH,

    push_to_hub=False,
    dataloader_pin_memory=True,

    bf16=True
)

trainer = Seq2SeqTrainer(
   model=model,
   args=training_args,
   train_dataset=tokenized_dataset["train"],
   eval_dataset=tokenized_dataset["test"],
   processing_class=tokenizer,
   data_collator=data_collator,
   compute_metrics=compute_metrics
)

trainer.train()

trainer.save_model(OUTPUT_DIR + "/final_model")
tokenizer.save_pretrained(OUTPUT_DIR + "/final_model")

print("===== Training Finished =====")
print("Final model saved to:", OUTPUT_DIR + "/final_model")