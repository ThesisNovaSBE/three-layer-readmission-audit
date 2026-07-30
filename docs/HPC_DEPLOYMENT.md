# HPC Deployment Guide — GWDG KISSKI Cluster

Stage 2 training (Clinical-Longformer, 249k notes) requires a GPU with ≥16 GB VRAM. This guide documents the reference deployment on GWDG KISSKI, an AI/ML-specific HPC cluster provided through the German National Research Data Infrastructure (NFDI).

## Cluster details

| Property | Value |
|----------|-------|
| Cluster | GWDG KISSKI |
| Login node | `glogin-gpu.hpc.gwdg.de` |
| Username | `u29346` |
| Project directory | `/projects/extern/kisski/kisski-nova-rpcl/dir.project` |
| GPU (requested) | NVIDIA A100-SXM4-80GB |
| Slurm partition | `kisski` |
| Scheduler | Slurm |
| Job script | `train_stage2.sh` |

## Prerequisites (one-time setup)

### 1. SSH config (`~/.ssh/config` on your Mac)

```
Host kisski
    HostName glogin-gpu.hpc.gwdg.de
    User u29346
    IdentityFile ~/.ssh/id_ed25519
```

After adding this, connect with `ssh kisski`.

### 2. Clone the repo on the cluster

```bash
ssh kisski
git clone <repo-url> ~/thesis
```

### 3. Create the conda environment

```bash
module load miniforge3
conda create -n thesis-env python=3.11 -y
conda activate thesis-env

# PyTorch with CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install -r ~/thesis/requirements.txt
```

### 4. Create the logs directory

```bash
mkdir -p /projects/extern/kisski/kisski-nova-rpcl/dir.project/logs
```

### 5. Configure MIMIC data paths

Create `~/thesis/.env`:

```bash
nano ~/thesis/.env
```

```
MIMIC_IV_DIR=/projects/extern/kisski/kisski-nova-rpcl/dir.project/mimic-iv
MIMIC_IV_NOTE_DIR=/projects/extern/kisski/kisski-nova-rpcl/dir.project/mimic-iv-note
```

## Data transfer (Mac → cluster)

**Always run rsync from your Mac terminal, not from inside the SSH session.**

The SSH alias `kisski` is defined in your Mac's `~/.ssh/config` and is not reachable from the cluster itself.

### Transfer MIMIC-IV structured tables (~9.9 GB)

```bash
rsync -av --progress \
  /path/to/local/mimic-iv/ \
  kisski:/projects/extern/kisski/kisski-nova-rpcl/dir.project/mimic-iv/
```

### Transfer MIMIC-IV-Note (~1.8 GB)

```bash
rsync -av --progress \
  /path/to/local/mimic-iv-note/ \
  kisski:/projects/extern/kisski/kisski-nova-rpcl/dir.project/mimic-iv-note/
```

### Transfer the Stage 1 artifact (~4.4 MB)

```bash
rsync -av \
  /path/to/local/20260401_Thesis/models/stage1_xgboost.joblib \
  kisski:~/thesis/models/stage1_xgboost.joblib
```

### Transfer updated code

```bash
rsync -av --exclude='.git' --exclude='data/' --exclude='models/' --exclude='.venv/' \
  /path/to/local/20260401_Thesis/ \
  kisski:~/thesis/
```

## Submitting the training job

```bash
ssh kisski
tmux new -s thesis           # or: tmux attach -t thesis
cd ~/thesis
sbatch train_stage2.sh
```

tmux is required so the session persists if your SSH connection drops.

## Monitoring

```bash
# Job queue
squeue -u u29346

# Live log
tail -f /projects/extern/kisski/kisski-nova-rpcl/dir.project/logs/stage2_<JOBID>.log

# Error log (check if job crashed)
cat /projects/extern/kisski/kisski-nova-rpcl/dir.project/logs/stage2_<JOBID>.err
```

If the job shows `R` in `squeue`, it is running. `PD` means pending (waiting for a free GPU node).

## Crash recovery

The job script checkpoints every 500 steps (≈12 min at batch_size=8, 2048 tokens, A100 80GB). If the job hits the 48h wall-time limit or is killed, **resubmit without any changes**:

```bash
sbatch train_stage2.sh
```

`train_stage2.py` uses `_find_resume_checkpoint()` to detect the latest `checkpoint-*` directory and passes it to `Trainer.train(resume_from_checkpoint=...)` automatically.

## Retrieving results (cluster → Mac)

After training completes, transfer the model weights back:

```bash
# From Mac terminal
rsync -av \
  kisski:~/thesis/models/ \
  /path/to/local/20260401_Thesis/models/
```

**MIMIC data must NOT be transferred to the Mac repository and must never be committed.**

## Post-training steps (run locally on Mac)

```bash
# Calibrate per-age-group thresholds
python -m src.stage2.calibrate

# Evaluate Stage 2 on test set
python -m src.stage2.evaluate

# Update MODEL_CARD.md with new metrics, then run Stage 3
python setup_stage3.py --limit 50
```

## Troubleshooting

| Symptom | Likely cause | Fix |
|---------|-------------|-----|
| Job exits in <30s, ExitCode 0 | `set -e` missing — Python error hidden | Add `set -e` + `set -o pipefail` to job script |
| `TypeError: 'AppConfig' object is not subscriptable` | Pydantic v2 dict access | Use `cfg.field` not `cfg["field"]` |
| CUDA not available / no GPU in nvidia-smi | `CUDA_VISIBLE_DEVICES=""` set unconditionally | Wrap in `if sys.platform == "darwin":` |
| Stage 1 artifact not found | `models/` not rsynced | Transfer `stage1_xgboost.joblib` separately |
| rsync times out from cluster | Running rsync inside SSH session | Run rsync from your Mac, not from the cluster |
| Permission denied on `/scratch/u29346/` | KISSKI doesn't pre-create `/scratch/` | Use `$PROJECT_DIR` instead |
| All nodes busy (0 free) | KISSKI is shared; jobs queue | Submit and wait; check `squeue` periodically |

## config.yaml settings for A100 80GB

```yaml
stage2:
  batch_size: 8
  gradient_accumulation_steps: 2   # effective batch size = 16
  bf16: true
  fp16: false
  gradient_checkpointing: false
  dataloader_num_workers: 8
```

For other hardware, see the comment block in `config.yaml` under "GPU / precision".
