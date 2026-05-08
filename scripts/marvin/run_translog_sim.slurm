#!/usr/bin/env bash
# SLURM batch script for Marvin (marvin.hpc.uni-bonn.de) GPU partition.
#
# Runs the translog AF simulation sweep using pixi-managed envs.
# Spawns four sweep workers (one per GPU) on the n=500 panel, plus a
# fifth process for the small n=2000 cell on GPU 0.
#
# Layout assumption:
#   $HOME/skillmodels-applications/                    # parent workspace (gitlab)
#   $HOME/skillmodels-applications/skillmodels/        # this repo (af-estimator branch)
#   $HOME/skillmodels-applications/sim_repro/          # sim runner code (rsync'd)
#   $HOME/sciebo_data/Skill estimation/Simulations/    # MATLAB results data
#
# Submit with:
#   sbatch scripts/marvin/run_translog_sim.slurm

#SBATCH --job-name=skillmodels-translog-sim
#SBATCH --account=ag_iame_gaudecker
#SBATCH --partition=sgpu_short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=4
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=03:30:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=hmgaudecker@gmail.com
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

module load Pixi
cd "$SKILLMODELS_ROOT"

nvidia-smi --list-gpus

# ---------------------------------------------------------------
# Launch four sweep workers, one per H100, plus a fifth for n=2000.
# ---------------------------------------------------------------
launch_worker() {
    local gpu_id="$1"
    local variant="$2"
    local n="$3"
    local start="$4"
    local count="$5"
    CUDA_VISIBLE_DEVICES="$gpu_id" pixi run -e tests-cuda12 python "$SIM_REPRO_ROOT/sim_sweep.py" --variant "$variant" --n "$n" --start "$start" --count "$count" --n-halton 10000 > "logs/sweep_${variant}_n${n}_gpu${gpu_id}_${SLURM_JOB_ID}.log" 2>&1 &
}

# Translog n=500 AF: split 500 sims across all 4 GPUs (125 each).
for gpu_id in 0 1 2 3; do
    launch_worker "$gpu_id" translog 500 $((125 * gpu_id)) 125
done

# Translog n=2000 AF: small cell (the .mat file holds 5 stored sims).
CUDA_VISIBLE_DEVICES=0 pixi run -e tests-cuda12 python "$SIM_REPRO_ROOT/sim_sweep.py" --variant translog --n 2000 --count 5 --n-halton 10000 > "logs/sweep_translog_n2000_gpu0_${SLURM_JOB_ID}.log" 2>&1 &

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
