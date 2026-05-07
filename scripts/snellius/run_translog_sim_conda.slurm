#!/usr/bin/env bash
# SLURM batch script for Snellius `gpu_h100` partition (Mamba/conda variant).
#
# Same job as `run_translog_sim.slurm` but launches workers via a
# pre-created conda env (`tests-cuda12`) instead of pixi. Use this when
# pixi is not available on the cluster. Create the env once on a login
# node:
#
#   module load 2024 && module load Mamba/24.9.0-0
#   mamba env create -f $HOME/skillmodels-applications/environment.yml
#   mamba activate tests-cuda12
#   pip install --upgrade --force-reinstall \
#     git+https://github.com/optimagic-dev/optimagic.git@probability-allow-fixed-entries
#
# Submit with:
#   sbatch scripts/snellius/run_translog_sim_conda.slurm

#SBATCH --job-name=skillmodels-translog-sim
#SBATCH --partition=gpu_h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=4
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=05:00:00
#SBATCH --output=logs/translog-sim_%j.out
#SBATCH --error=logs/translog-sim_%j.err

set -euo pipefail

# ---------------------------------------------------------------
# Environment
# ---------------------------------------------------------------
SKILLMODELS_ROOT="${SKILLMODELS_ROOT:-$HOME/skillmodels-applications/skillmodels}"
SIM_REPRO_ROOT="${SIM_REPRO_ROOT:-$HOME/skillmodels-applications/sim_repro}"
export SIM_RESULTS_DIR="${SIM_RESULTS_DIR:-$HOME/sciebo_data/Skill estimation/Simulations/Results}"
export SIM_REPRO_OUT="${SIM_REPRO_OUT:-$SIM_REPRO_ROOT/estimates}"

mkdir -p logs "$SIM_REPRO_OUT"

# Point at the pre-created conda env directly. We deliberately skip
# `conda activate` because the Mamba module loaded on the H100 nodes
# ships without `etc/profile.d/conda.sh` (login-node and compute-node
# arch dirs differ on Snellius). Setting CONDA_PREFIX + PATH is enough
# for the env's interpreter and entry points.
export CONDA_PREFIX="${CONDA_PREFIX_OVERRIDE:-$HOME/.conda/envs/tests-cuda12}"
export PATH="$CONDA_PREFIX/bin:$PATH"

cd "$SKILLMODELS_ROOT"

# Sanity check: every GPU visible.
nvidia-smi --list-gpus

# ---------------------------------------------------------------
# Launch four sweep workers, one per H100, plus a fifth for n=2000.
# Each worker handles a disjoint sim slice via --start / --count.
# ---------------------------------------------------------------
launch_worker() {
    local gpu_id="$1"
    local variant="$2"
    local n="$3"
    local start="$4"
    local count="$5"
    CUDA_VISIBLE_DEVICES="$gpu_id" "$CONDA_PREFIX/bin/python" \
        "$SIM_REPRO_ROOT/sim_sweep.py" \
        --variant "$variant" --n "$n" --start "$start" --count "$count" \
        --n-halton 10000 \
        > "logs/sweep_${variant}_n${n}_gpu${gpu_id}_${SLURM_JOB_ID}.log" 2>&1 &
}

launch_chs_worker() {
    local gpu_id="$1"
    local variant="$2"
    local n="$3"
    local start="$4"
    local count="$5"
    CUDA_VISIBLE_DEVICES="$gpu_id" "$CONDA_PREFIX/bin/python" \
        "$SIM_REPRO_ROOT/sim_sweep_chs.py" \
        --variant "$variant" --n "$n" --start "$start" --count "$count" \
        > "logs/sweep_chs_${variant}_n${n}_gpu${gpu_id}_${SLURM_JOB_ID}.log" 2>&1 &
}

# Translog n=500 AF: split 500 sims across all 4 GPUs (125 each).
for gpu_id in 0 1 2 3; do
    launch_worker "$gpu_id" translog 500 $((125 * gpu_id)) 125
done

# Translog n=500 CHS: same dataset + same normalisations, comparison
# point against AF.
for gpu_id in 0 1 2 3; do
    launch_chs_worker "$gpu_id" translog 500 $((125 * gpu_id)) 125
done

# Translog n=2000: small cell (the .mat file holds 5 stored sims).
CUDA_VISIBLE_DEVICES=0 "$CONDA_PREFIX/bin/python" \
    "$SIM_REPRO_ROOT/sim_sweep.py" \
    --variant translog --n 2000 --count 5 --n-halton 10000 \
    > "logs/sweep_translog_n2000_gpu0_${SLURM_JOB_ID}.log" 2>&1 &
CUDA_VISIBLE_DEVICES=0 "$CONDA_PREFIX/bin/python" \
    "$SIM_REPRO_ROOT/sim_sweep_chs.py" \
    --variant translog --n 2000 --count 5 \
    > "logs/sweep_chs_translog_n2000_gpu0_${SLURM_JOB_ID}.log" 2>&1 &

wait

echo "All workers exited; aggregating results..."
"$CONDA_PREFIX/bin/python" - <<'PY'
import pickle
from pathlib import Path
import os

root = Path(os.environ["SIM_REPRO_OUT"])
for cell in (
    "translog_n500",
    "translog_n500_chs",
    "translog_n2000",
    "translog_n2000_chs",
):
    if not (root / cell).exists():
        continue
    pkls = sorted((root / cell).glob("sim_*.pkl"))
    ok = sum(1 for f in pkls if pickle.load(open(f, "rb")).get("success"))
    fail = len(pkls) - ok
    print(f"{cell}: {ok} ok, {fail} failed (out of {len(pkls)})")
PY
