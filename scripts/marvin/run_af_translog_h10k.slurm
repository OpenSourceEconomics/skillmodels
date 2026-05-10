#!/usr/bin/env bash
# AF-only translog sweep at n_halton=10000 (matches MATLAB AF reference).
# Both Stage-A-only and Stage-A+B variants on 4 A100s. CHS is not
# re-run here: 500/500 CHS-min results from job 25928963 already cover it.

#SBATCH --job-name=skillmodels-af-translog-h10k
#SBATCH --account=ag_iame_gaudecker
#SBATCH --partition=sgpu_short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=4
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=08:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=hmgaudecker@gmail.com
#SBATCH --output=logs/af-translog-h10k_%j.out
#SBATCH --error=logs/af-translog-h10k_%j.err

set -euo pipefail

SKILLMODELS_ROOT="${SKILLMODELS_ROOT:-$HOME/skillmodels-applications/skillmodels}"
SIM_REPRO_ROOT="${SIM_REPRO_ROOT:-$HOME/skillmodels-applications/sim_repro}"
export SIM_RESULTS_DIR="${SIM_RESULTS_DIR:-$HOME/sciebo_data/Skill estimation/Simulations/Results}"
# New subroot so we do not collide with existing h=2000 cells.
export SIM_REPRO_OUT="${SIM_REPRO_OUT:-$SIM_REPRO_ROOT/estimates_af_h10000}"

mkdir -p logs "$SIM_REPRO_OUT"

# User-installed pixi (AMD64-generic upstream); never `module load Pixi`.
export PATH="$HOME/.pixi/bin:$PATH"
cd "$SKILLMODELS_ROOT"
echo "Using pixi: $(which pixi) $(pixi --version)"

nvidia-smi --list-gpus

N_HALTON=10000
VARIANT=translog
N=500

launch_af_worker() {
    local gpu_id="$1"
    local stage_flag="$2"
    local stage_tag="$3"
    local start="$4"
    local count="$5"
    CUDA_VISIBLE_DEVICES="$gpu_id" \
        pixi run -e tests-cuda13 python "$SIM_REPRO_ROOT/sim_sweep.py" \
        --variant "$VARIANT" \
        --n "$N" \
        --start "$start" \
        --count "$count" \
        --n-halton "$N_HALTON" \
        --out-suffix "_h${N_HALTON}" \
        $stage_flag \
        > "logs/af-h10k_${stage_tag}_n${N}_gpu${gpu_id}_${SLURM_JOB_ID}.log" 2>&1 &
}

# Stage A: GPUs 0,1, 250 sims each.
launch_af_worker 0 "--no-two-stage-measurement" "stagea"   0 250
launch_af_worker 1 "--no-two-stage-measurement" "stagea" 250 250

# Stage A+B: GPUs 2,3, 250 sims each.
launch_af_worker 2 "--two-stage-measurement"    "stageab"   0 250
launch_af_worker 3 "--two-stage-measurement"    "stageab" 250 250

wait

echo
echo "All workers exited; computing per-cell coverage..."
pixi run -e tests-cuda13 python - <<'PY'
import os
import pickle
from pathlib import Path

root = Path(os.environ["SIM_REPRO_OUT"])
cells = (
    "translog_n500_stagea_h10000",
    "translog_n500_stageab_h10000",
)
for cell in cells:
    if not (root / cell).exists():
        print(f"{cell}: MISSING")
        continue
    pkls = sorted((root / cell).glob("sim_*.pkl"))
    ok, fail = 0, 0
    for f in pkls:
        with f.open("rb") as fh:
            payload = pickle.load(fh)
        if payload.get("success"):
            ok += 1
        else:
            fail += 1
    print(f"{cell}: {ok} ok, {fail} failed (out of {len(pkls)})")
PY
