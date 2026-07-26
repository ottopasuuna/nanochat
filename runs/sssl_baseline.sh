#!/bin/bash

# MSA (MiniMax Sparse Attention) smoke run, modeled on runs/quick.sh (~10 min on one GPU).
# Same size/data as quick.sh so losses/bpb are comparable to the SSSL baseline.
# Config: all-MSA layers (final layer forced to dense L), budget k*B_k = 4*64 = 256 of
# 1024 tokens (4x sparsity), 5% indexer warmup. Watch: LM loss continuity at the
# warmup->sparse switch, KL decreasing, block recall rising above chance (0.25).

export OMP_NUM_THREADS=1
export NANOCHAT_BASE_DIR="$HOME/.cache/nanochat"
mkdir -p $NANOCHAT_BASE_DIR

command -v uv &> /dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
[ -d ".venv" ] || uv venv
uv sync --extra gpu
source .venv/bin/activate

if [ -z "$WANDB_RUN" ]; then
    WANDB_RUN=dummy
fi

MODEL_TAG=sssl_baseline
torchrun --standalone --nnodes 1 --nproc_per_node=1 -m scripts.base_train -- \
  --depth=4 \
  --aspect-ratio=64 \
  --head-dim=128 \
  --window-pattern=SSSL \
  --max-seq-len=1024 \
  --device-batch-size=8 \
  --eval-every=-1 \
  --core-metric-every=-1 \
  --sample-every=-1 \
  --save-every=-1 \
  --target-flops=5e15 \
  --num-iterations=-1 \
  --run=$WANDB_RUN \
  --log-interval=50 \
  --model-tag=$MODEL_TAG \
  --device-type=cuda

python -m scripts.base_eval \
    --eval bpb,sample \
    --split-tokens 2048 \
    --device-batch-size=1 \
    --model-tag $MODEL_TAG \
    --device-type cuda
