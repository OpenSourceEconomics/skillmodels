#!/usr/bin/env bash
# SLURM batch script for Snellius `gpu_h100` partition.
#
# Runs the translog AF simulation sweep on one H100 node (4 GPUs).
# Spawns four independent sweep processes — one per GPU — each running a
# disjoint slice of the 500-sim panel, plus a fifth process for the
# (small) n=2000 cell on GPU 0.
#
# Layout assumption:
#   $HOME/skillmodels-applications/skillmodels/        # this repo
#   $HOME/skillmodels-applications/sim_repro/          # sim runner code
#   $HOME/sciebo_data/Skill estimation/Simulations/    # MATLAB results data
#
# Submit with:
#   sbatch scripts/snellius/run_translog_sim.slurm
#
# Per-sim wall time on H100 is roughly 60-90 s at n=500 / 10k Halton
# (vs. ~480 s on the local RTX 3070), so the 500 + 5 sims should finish
# well inside the 24 h wall clock.

#SBATCH --job-name=skillmodels-translog-sim
#SBATCH --partition=gpu_h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=4
#SBATCH --cpus-per-task=64
#SBATCH --mem=384G
#SBATCH --time=03:30:00
#SBATCH --output=logs/translog-sim_%j.out
#SBATCH --error=logs/translog-sim_%j.err

set -euo pipefail

# ---------------------------------------------------------------
# Environment
# ---------------------------------------------------------------
SKILLMODELS_ROOT="${SKILLMODELS_ROOT:-$HOME/skillmodels-applications/skillmodels}"
SIM_REPRO_ROOT="${SIM_REPRO_ROOT:-$HOME/skillmodels-applications/sim_repro}"
# Path on Snellius where the MATLAB result files live; copy from your
# local sciebo (e.g. via rsync) before submitting.
export SIM_RESULTS_DIR="${SIM_RESULTS_DIR:-$HOME/sciebo_data/Skill estimation/Simulations/Results}"
export SIM_REPRO_OUT="${SIM_REPRO_OUT:-$SIM_REPRO_ROOT/estimates}"

mkdir -p logs "$SIM_REPRO_OUT"

# Pixi installs JAX + the cuda12 stack from conda-forge. The script
# expects `pixi install -e tests-cuda12` to have been run on a login
# node before submission (compute nodes have no internet on Snellius).
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
    CUDA_VISIBLE_DEVICES="$gpu_id" pixi run -e tests-cuda12 python \
        "$SIM_REPRO_ROOT/sim_sweep.py" \
        --variant "$variant" --n "$n" --start "$start" --count "$count" \
        --n-halton 10000 \
        > "logs/sweep_${variant}_n${n}_gpu${gpu_id}_${SLURM_JOB_ID}.log" 2>&1 &
}

# Translog n=500 AF: split 500 sims across all 4 GPUs (125 each).
for gpu_id in 0 1 2 3; do
    launch_worker "$gpu_id" translog 500 $((125 * gpu_id)) 125
done

# Translog n=2000 AF: small cell (the .mat file holds 5 stored sims).
# Run on GPU 0 alongside its n=500 chunk; H100 has plenty of memory.
CUDA_VISIBLE_DEVICES=0 pixi run -e tests-cuda12 python \
    "$SIM_REPRO_ROOT/sim_sweep.py" \
    --variant translog --n 2000 --count 5 --n-halton 10000 \
    > "logs/sweep_translog_n2000_gpu0_${SLURM_JOB_ID}.log" 2>&1 &

wait

echo "All workers exited; aggregating results..."
pixi run -e tests-cuda12 python - <<'PY'
import pickle
from pathlib import Path
import os

root = Path(os.environ["SIM_REPRO_OUT"])
for cell in (
    "translog_n500",
    "translog_n2000",
):
    if not (root / cell).exists():
        continue
    pkls = sorted((root / cell).glob("sim_*.pkl"))
    ok = sum(1 for f in pkls if pickle.load(open(f, "rb")).get("success"))
    fail = len(pkls) - ok
    print(f"{cell}: {ok} ok, {fail} failed (out of {len(pkls)})")
PY
