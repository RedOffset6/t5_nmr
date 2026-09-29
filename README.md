# NMR-to-SMILES with FLAN-T5

This project fine-tunes Google's [FLAN-T5](https://huggingface.co/docs/transformers/model_doc/flan-t5) models to predict a molecule's structure, written as a SMILES string, from a text encoding of its NMR spectrum. The task is framed as text-to-text translation:

    input:  "predict SMILES from NMR spectrum: <NMR spectrum as text>"
    output: "<SMILES string>"

The best model so far, FLAN-T5-XL trained for 10 epochs, predicts the exact SMILES string for **88.2%** of the 79,441 test molecules.

## Workflow

    ┌────────────┐   ┌──────────────┐   ┌─────────────────────┐   ┌──────────────┐
    │ 1. Set up  │ → │ 2. Train     │ → │ 3. Evaluate         │ → │ 4. Results   │
    │ env + data │   │ start_       │   │ evaluate_*_array.sh │   │ reports/     │
    │            │   │ training_*.sh│   │ (4 parallel chunks) │   │              │
    └────────────┘   └──────┬───────┘   └──────────┬──────────┘   └──────────────┘
                            │                      │
                        t5_train.py      evaluate_exact_match.py
                            │                      │
                            └─→ outputs/<run>/final_model ─┘

Every step runs as a Slurm job on a GPU cluster.

### 1. Set up the environment and data

Create the Conda environment (Python 3.11, PyTorch 2.12 with CUDA 12.6, Transformers 5.12):

    conda env create -f environment.yml
    conda activate t5

Copy the dataset into `alberts_2d/` next to the scripts. It isn't stored in Git because of its size and possible redistribution restrictions. The folder needs six files:

    alberts_2d/
      src-train.txt   tgt-train.txt   # 679,213 pairs
      src-val.txt     tgt-val.txt     #  35,749 pairs
      src-test.txt    tgt-test.txt    #  79,441 pairs

Line *N* of each `src-*.txt` file is an NMR spectrum, and line *N* of the matching `tgt-*.txt` file is its SMILES string. Check your copy against the checksums in `data/manifest.tsv`:

    sha256sum alberts_2d/*.txt

Create the folder for Slurm logs. Slurm doesn't create it, and jobs fail without it:

    mkdir -p logs

Optionally, check that a GPU node works before using hours of GPU time:

    sbatch test_gpu.sh

### 2. Train a model

Pick a training script and submit it:

    sbatch start_training_xl_4x4_10ep.sh

The job:

1. Downloads the pretrained FLAN-T5 model from Hugging Face. The first run needs internet access.
2. Fine-tunes it on `alberts_2d/` training data, measuring validation loss after each epoch.
3. Saves a checkpoint every 2,000 steps to `outputs/<run>/checkpoint-*`, keeping the latest two.
4. Saves the finished model to `outputs/<run>/final_model`.
5. Scores exact match on the first 1,000 test molecules as a quick check, saved to `outputs/<run>/test_results.json`. This is not the official score; that comes from step 3.

**Long runs:** each job has a one-day time limit. If a job runs out of time, submit the same script again. It resumes from the latest checkpoint in its output folder.

### 3. Evaluate the model on the full test set

Submit the evaluation script with the same name as the training script:

    sbatch evaluate_xl_4x4_10ep_array.sh

This is a Slurm array job. It splits the 79,441 test molecules into 4 chunks of 20,000 and evaluates them in parallel on separate GPUs. Each chunk writes its score to `outputs/<run>/test_<start>_<end>_results.json`. While a chunk is running, it saves progress every 100 batches to a matching `.partial.json` file.

The model predicts with greedy decoding, and a prediction counts as correct only if it is **exactly** the same string as the reference SMILES.

To test a trained model on just 10 molecules first, run `sbatch evaluate_check.sh`.

### 4. Combine the results

Combine the four chunks, weighting each by its number of molecules:

    python -c "import json,glob,sys; r=[json.load(open(f)) for f in glob.glob(sys.argv[1]+'/test_*_results.json')]; print(sum(x['matches'] for x in r)/sum(x['samples'] for x in r))" outputs/flan-t5-xl_nmr_input1536_4x4_ep10

Record the result in `reports/evaluation_summary.tsv`.

## Scripts

### Python programs

| File | Purpose |
|---|---|
| `t5_train.py` | Training pipeline shared by every training script. It is configured entirely through environment variables (see below). |
| `evaluate_exact_match.py` | Loads a trained `final_model`, predicts SMILES for a range of test molecules, and scores exact match. Run `python evaluate_exact_match.py --help` for its options. |

### Training scripts

Training scripts are named `start_training_<model>_<batch>x<accumulation>_<epochs>.sh`. For example, `start_training_xl_4x4_10ep.sh` trains FLAN-T5-XL with 4 examples per step and 4 steps of gradient accumulation (16 examples per update) for 10 epochs.

| Model | Parameters | Scripts |
|---|---|---|
| small | 80M | `small_4x4_3ep` |
| base | 250M | `base_16x1_3ep` |
| large | 780M | `large_4x4_3ep`, `large_4x4_10ep` |
| xl | 3B | `xl_4x4_3ep`, `xl_1x16_3ep`, `xl_4x4_10ep`, `xl_1x16_10ep`, `xl_2x8_10ep` |
| xxl | 11B | `xxl_4x4_10ep`, `xxl_2x8_10ep`, `xxl_1x16_10ep` |

Every configuration updates the model with 16 examples at a time. Configurations with a smaller batch per step, such as 1×16, use less GPU memory but train more slowly.

Each script sets environment variables and runs `t5_train.py`. To create a new configuration, copy a script, change the variables, and give it a new `OUTPUT_DIR`. If two scripts share an output folder, one run will resume from the other's checkpoints.

| Variable | Default | Meaning |
|---|---|---|
| `MODEL_NAME` | `google/flan-t5-base` | Hugging Face model to fine-tune |
| `DATA_DIR` | `alberts_2d/` | Folder with the `src-*`/`tgt-*` files |
| `OUTPUT_DIR` | `outputs/<model>_nmr` | Where checkpoints and the final model go |
| `TRAIN_BATCH_SIZE` | 16 | Examples per GPU per step |
| `GRAD_ACCUMULATION_STEPS` | 1 | Steps combined into each model update |
| `NUM_EPOCHS` | 3 | Passes over the training data |
| `LEARNING_RATE` | 5e-5 | Optimizer learning rate |
| `WEIGHT_DECAY` | 0.01 | Optimizer weight decay |
| `GRADIENT_CHECKPOINTING` | off | Trades speed for lower GPU memory |
| `TARGET_MAX_LENGTH` | 128 | SMILES strings longer than this many tokens are cut off |
| `EVAL_BATCH_SIZE` | 32 | Batch size for validation loss |
| `SAVE_STEPS` / `SAVE_TOTAL_LIMIT` | 2000 / 2 | Checkpoint frequency and how many to keep |
| `TEST_SAMPLE_SIZE` | 1000 | Test molecules in the quick end-of-training check |
| `GENERATION_BATCH_SIZE` | 32 | Batch size for the quick check |
| `GENERATION_MAX_NEW_TOKENS` | 128 | Longest SMILES the quick check can generate |
| `SEED` | 42 | Random seed |
| `LOCAL_FILES_ONLY` | 0 | Set to 1 on offline nodes after the model is downloaded once |

NMR inputs are never truncated. T5 has no fixed maximum input length, so the whole spectrum is always used. Very long inputs use a lot of GPU memory; if a job runs out of memory, use a configuration with a smaller batch per step.

### Evaluation scripts

| File | Purpose |
|---|---|
| `evaluate_<model>_<batch>x<accumulation>_<epochs>_array.sh` | Evaluates the model trained by the training script with the same name. Scripts exist for `small_4x4_3ep`, `base_16x1_3ep`, `large_4x4_3ep`, `xl_4x4_3ep`, `xl_1x16_3ep` and `xl_4x4_10ep`. |
| `evaluate_check.sh` | Evaluates 10 molecules to confirm a model loads and runs |
| `test_gpu.sh` | Prints GPU and PyTorch CUDA information for a cluster node |

To evaluate another run, copy an existing evaluation script and change `--model-path` and the job name.

## Results

Exact match on all 79,441 test molecules, from `reports/evaluation_summary.tsv`. These runs used earlier versions of the scripts, which truncated inputs to the length shown. Current scripts don't truncate inputs, so new runs may score differently.

| Model | 512 tokens | 1024 tokens | 1536 tokens |
|---|---|---|---|
| small, 3 epochs | 15.9% | 29.9% | 36.7% |
| base, 3 epochs | – | 65.1% | 68.9% |
| large, 3 epochs | – | 78.6% | 80.6% |
| xl, 3 epochs | – | 83.8% | 84.9–85.2% |
| **xl, 10 epochs** | – | – | **88.2%** |

Accuracy improves with model size, with longer inputs and with longer training. The large (10 epochs) and XXL runs have not been evaluated.

`reports/evaluation_chunks.tsv` has the score for each chunk. `eval_check_5928345` in the summary is a 10-molecule smoke test and isn't comparable with the full evaluations.

## Repository layout

    t5_train.py                  training pipeline
    evaluate_exact_match.py      evaluation program
    start_training_*.sh          Slurm training jobs
    evaluate_*_array.sh          Slurm evaluation jobs
    evaluate_check.sh            10-molecule evaluation smoke test
    test_gpu.sh                  GPU check
    environment.yml              Conda environment
    environment.txt              full package list of the environment
    data/                        dataset manifest and checksums
    reports/                     evaluation results

These folders are created locally and aren't stored in Git: `alberts_2d/` (dataset), `outputs/` (models and checkpoints) and `logs/` (Slurm logs). `nmr_expt_data/` is a second dataset listed in `data/manifest.tsv`; the current scripts don't use it.

## Running on another cluster

The Slurm scripts are set up for the original cluster. Before running them elsewhere, change:

- `#SBATCH --partition=workq` to a GPU partition on your cluster
- `module load cuda/12.6` to your cluster's CUDA module
- `source /home/b5an/jucloud.b5an/miniforge3/bin/activate` to your own Conda installation
