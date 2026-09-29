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

Before the first multi-GPU run, check that all 4 GPUs of a node can communicate (see [Multi-GPU training](#multi-gpu-training)):

    sbatch test_multi_gpu.sh

### 2. Train a model

Pick a training script and submit it:

    sbatch start_training_xl_4x1x4_10ep.sh

The job:

1. Downloads the pretrained FLAN-T5 model from Hugging Face. The first run needs internet access.
2. Tokenizes the data once and caches it in `outputs/<run>/tokenized/`, so a resubmitted job starts faster. Then it fine-tunes the model on `alberts_2d/` training data, measuring validation loss after each epoch. Scripts with a GPU count in their name train on all 4 GPUs of a node, or on 8 GPUs across two nodes.
3. Saves a checkpoint every 2,000 steps to `outputs/<run>/checkpoint-*`, keeping the latest two.
4. Saves the finished model to `outputs/<run>/final_model`.
5. Scores exact match on the first 1,000 test molecules as a quick check, saved to `outputs/<run>/test_results.json`. This is not the official score; that comes from step 3.

**Long runs:** each job has a one-day time limit. If a job runs out of time, submit the same script again. It resumes from the latest checkpoint in its output folder. A multi-GPU run can only resume on the same number of GPUs, with the same `PARALLEL_MODE`, as the job that wrote the checkpoint, so don't edit the script's resources between resubmissions.

**Smoke test:** to test a configuration end to end before a long run, submit it with `MAX_STEPS` set, for example `sbatch --export=ALL,MAX_STEPS=50 start_training_xxl_2x2x4_10ep.sh`. `sbatch --export` sets the variable before the script's own `export` lines run, so this works because the scripts don't set `MAX_STEPS`. Delete the smoke test's output folder afterwards, or the real run will resume from its checkpoints.

### 3. Evaluate the model on the full test set

Submit the evaluation script with the same name as the training script:

    sbatch evaluate_xl_4x4_10ep_array.sh

This is a Slurm array job. It splits the 79,441 test molecules into 4 chunks of 20,000 and evaluates them in parallel on separate GPUs. Each chunk writes two files to `outputs/<run>/`:

- `test_<start>_<end>_results.json`: top-1 to top-N exact-match scores. While a chunk is running, it saves progress every 100 batches to a matching `.partial.json` file.
- `test_<start>_<end>_predictions.txt`: the model's N most likely SMILES for each spectrum, one per line, best first. Spectrum *i* in the chunk occupies lines *i*·N + 1 to (*i* + 1)·N. This is the same layout as `prd-test.txt` from earlier models.

**Number of outputs per spectrum:** set `NUM_OUTPUTS` near the top of the evaluation script (default 10). The script passes it to `evaluate_exact_match.py --num-outputs`.

- `NUM_OUTPUTS=1` uses greedy decoding and only reports top-1.
- `NUM_OUTPUTS=N` (N > 1) uses beam search with N beams and returns the N highest-scoring SMILES. Top-*n* accuracy counts a spectrum as correct if the reference SMILES is anywhere in the first *n* outputs. Beam search can also change the top-1 prediction, so its top-1 score may differ slightly from greedy decoding.
- More outputs make evaluation slower and use more GPU memory. If a chunk runs out of memory or time, lower `NUM_OUTPUTS` or `--batch-size`, or raise `#SBATCH --time`.

A prediction counts as correct only if it is **exactly** the same string as the reference SMILES.

To test a trained model on just 10 molecules first, run `sbatch evaluate_check.sh`.

### 4. Combine the results

Combine the four chunks' top-1 to top-N scores, weighting each chunk by its number of molecules:

    python -c "import json,glob,sys; r=[json.load(open(f)) for f in glob.glob(sys.argv[1]+'/test_*_results.json')]; n=sum(x['samples'] for x in r); t=[sum(c) for c in zip(*(x.get('top_n_matches',[x['matches']]) for x in r))]; [print(f'top-{k}: {c/n:.4f}') for k,c in enumerate(t,1)]" outputs/flan-t5-xl_nmr_input1536_4x4_ep10

Join the four prediction files, in order, into one file for the whole test set:

    cd outputs/flan-t5-xl_nmr_input1536_4x4_ep10
    cat test_0_20000_predictions.txt test_20000_40000_predictions.txt \
        test_40000_60000_predictions.txt test_60000_79441_predictions.txt > prd-test.txt

Record the result in `reports/evaluation_summary.tsv`.

## Scripts

### Python programs

| File | Purpose |
|---|---|
| `t5_train.py` | Training pipeline shared by every training script. It is configured entirely through environment variables (see below). |
| `evaluate_exact_match.py` | Loads a trained `final_model`, predicts the N most likely SMILES (`--num-outputs`) for a range of test molecules, scores top-1 to top-N exact match, and saves the predictions. Run `python evaluate_exact_match.py --help` for its options. |

### Training scripts

Training scripts are named `start_training_<model>_<batch>x<accumulation>x<gpus>_<epochs>.sh`. For example, `start_training_xxl_2x2x4_10ep.sh` trains FLAN-T5-XXL with 2 examples per GPU per step, 2 steps of gradient accumulation and 4 GPUs (2 × 2 × 4 = 16 examples per update) for 10 epochs. Older single-GPU scripts leave out the GPU count: `start_training_xl_4x4_10ep.sh` means 4 × 4 on one GPU.

| Model | Parameters | 1 GPU | 4 GPUs (1 node) | 8 GPUs (2 nodes) |
|---|---|---|---|---|
| small | 80M | `small_4x4_3ep` | `small_4x1x4_3ep` | |
| base | 250M | `base_16x1_3ep` | | |
| large | 780M | `large_4x4_3ep`, `large_4x4_10ep` | `large_4x1x4_10ep` | |
| xl | 3B | `xl_4x4_3ep`, `xl_1x16_3ep`, `xl_4x4_10ep`, `xl_1x16_10ep`, `xl_2x8_10ep` | `xl_4x1x4_10ep` | |
| xxl | 11B | none work (see [Multi-GPU training](#multi-gpu-training)) | `xxl_2x2x4_10ep`, `xxl_1x4x4_10ep` | `xxl_2x1x8_10ep` |

Every configuration updates the model with 16 examples at a time: batch per GPU × accumulation steps × number of GPUs = 16. This keeps new runs comparable with the 88.2% XL result. Configurations with a smaller batch per step, such as 1×16, use less GPU memory but train more slowly. The old single-GPU `xxl_*` scripts are kept for reference but always run out of memory.

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
| `PARALLEL_MODE` | `none`, or `ddp` under torchrun | `none`: one GPU. `ddp`: every GPU holds a full copy of the model. `fsdp`: the model is split across GPUs. See [Multi-GPU training](#multi-gpu-training) |
| `DATALOADER_NUM_WORKERS` | 4 | Worker processes per GPU that prepare batches; also used to tokenize the data |
| `MAX_STEPS` | 0 | Stop after this many model updates, for smoke tests. 0 trains for `NUM_EPOCHS` |
| `TOKENIZED_CACHE_DIR` | `<OUTPUT_DIR>/tokenized` | Where the tokenized dataset is cached. It is rebuilt when the model, prefix, `TARGET_MAX_LENGTH` or `DATA_DIR` changes |

NMR inputs are never truncated. T5 has no fixed maximum input length, so the whole spectrum is always used. Very long inputs use a lot of GPU memory; if a job runs out of memory, use a configuration with a smaller batch per step.

### Evaluation scripts

| File | Purpose |
|---|---|
| `evaluate_<configuration>_array.sh` | Evaluates the model trained by the training script with the same configuration name. Scripts exist for `small_4x4_3ep`, `base_16x1_3ep`, `large_4x4_3ep`, `xl_4x4_3ep`, `xl_1x16_3ep` and `xl_4x4_10ep`, and for every multi-GPU configuration: `small_4x1x4_3ep`, `large_4x1x4_10ep`, `xl_4x1x4_10ep`, `xxl_2x2x4_10ep`, `xxl_1x4x4_10ep` and `xxl_2x1x8_10ep`. |
| `evaluate_check.sh` | Evaluates 10 molecules to confirm a model loads and runs |
| `test_gpu.sh` | Prints GPU and PyTorch CUDA information for a cluster node |
| `test_multi_gpu.sh` | Starts one process per GPU with torchrun, checks that they can communicate, and measures the bandwidth between them. `sbatch --nodes=2 test_multi_gpu.sh` checks the network between two nodes |

To evaluate another run, copy an existing evaluation script and change `--model-path` and the job name. Evaluation always uses one GPU per chunk, whichever way the model was trained. Even XXL fits on one GPU for evaluation, because it is loaded in bf16 (about 23 GB).

## Multi-GPU training

An Isambard-AI node has 4 GH200 GPUs, each with 96 GB of memory, connected by NVLink. Multi-GPU scripts start one training process per GPU with `torchrun`, request all 4 GPUs so the job never shares a node, and set `PARALLEL_MODE`.

**Why XXL needs several GPUs.** Full fine-tuning with the AdamW optimizer in mixed precision keeps about 16 bytes per parameter on the GPU: fp32 weights (4), fp32 gradients (4) and two optimizer moments (8). This is before counting activations, the intermediate results kept for the backward pass.

| Model | Training state | 1 GPU | Split over 4 GPUs | Split over 8 GPUs |
|---|---|---|---|---|
| large (0.78B) | ~12 GB | fits | – | – |
| xl (2.85B) | ~46 GB | fits only with gradient checkpointing | ~12 GB | ~6 GB |
| xxl (11.3B) | ~180 GB | never fits | ~45 GB | ~23 GB |

No batch size or gradient checkpointing setting makes XXL fit on one 96 GB GPU.

**Which mode to use:**

- `ddp` (data parallel), for small, base and large: every GPU holds a full copy of the model and processes different examples. The GPUs average their gradients after each step. Training is about 4 times faster than on one GPU.
- `fsdp` (fully sharded data parallel), for xl and xxl: the weights, gradients and optimizer state are split across the GPUs. Each GPU briefly gathers one T5 block's weights when it needs them. This is the same optimization as a single-GPU run, just spread over more memory. XL no longer needs gradient checkpointing. XXL still uses it to leave room for activations.

If `xxl_2x2x4_10ep` runs out of GPU memory, use `xxl_1x4x4_10ep`, or `xxl_2x1x8_10ep` on two nodes.

**Checkpoints.** An FSDP checkpoint is stored as one shard per GPU. A job can therefore only resume from a checkpoint written with the same `PARALLEL_MODE` and the same number of GPUs, and a checkpoint from a single-GPU run can't be resumed on 4 GPUs. The final model is gathered into a normal Hugging Face model in `final_model/`, which loads on any number of GPUs. For XXL, each checkpoint takes about 135 GB, because it includes the optimizer state. With `SAVE_TOTAL_LIMIT=2` and the 45 GB final model, one XXL run needs about 315 GB of project storage.

**Host memory.** The multi-GPU scripts request the whole node's memory (`--mem=0`) and all its CPU cores (`--cpus-per-task=288`, 4 × 72 Grace cores). Rank 0 loads the fp32 XXL weights (45 GB) and gathers the full model when saving, and the other processes wait. The tokenized dataset is memory-mapped, so all processes on a node share one copy. If your partition doesn't allow `--mem=0`, request about 360 GB.

**Two nodes.** Within a node, GPUs communicate over NVLink with no extra setup. Between nodes, NCCL needs the aws-ofi-nccl plugin to use the Slingshot network. `start_training_xxl_2x1x8_10ep.sh` and `test_multi_gpu.sh` load `brics/nccl` and `brics/aws-ofi-nccl` and set the variables from the [Isambard-AI NCCL guide](https://docs.isambard.ac.uk/user-documentation/guides/nccl/). Run `sbatch --nodes=2 test_multi_gpu.sh` first. With `NCCL_DEBUG=INFO`, the log should show `NET/AWS Libfabric`. If it shows `NET/Socket`, NCCL has fallen back to TCP, which works but is slow.

**Checking a new setup**, in order:

1. `sbatch test_multi_gpu.sh`: all 4 ranks print their GPU, and rank 0 prints `all_reduce OK` and the bandwidth.
2. Submit `small_4x1x4_3ep` with `MAX_STEPS=200` and `start_training_small_4x4_3ep.sh` with the same `MAX_STEPS`, and compare their training losses. They should track closely, but not exactly, because the examples are processed in a different order.
3. Submit `xl_4x1x4_10ep` with `MAX_STEPS=100`, cancel it after its first checkpoint, and resubmit it to check that an FSDP job can save and resume.
4. Submit `xxl_2x2x4_10ep` with `MAX_STEPS=50`. At the end, rank 0 prints `Peak GPU memory allocated (GB)`.

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
    test_multi_gpu.sh            multi-GPU and multi-node communication check
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

For multi-GPU scripts, also change:

- `#SBATCH --gres=gpu:4`, `GPUS_PER_NODE=4` and `--cpus-per-task` to match your nodes. Keep batch × accumulation × GPUs at 16.
- `--mem=0`, if your cluster doesn't allow whole-node memory requests. See the host memory note under [Multi-GPU training](#multi-gpu-training).
- In the two-node script, the `brics/*` modules and the `NCCL_*`/`FI_*` variables. These are specific to Isambard-AI's Slingshot network. On InfiniBand clusters NCCL usually needs no plugin, and on other networks check your cluster's NCCL documentation.
