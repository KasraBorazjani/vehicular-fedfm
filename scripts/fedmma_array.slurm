#!/bin/bash
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# FedMMA — SLURM job array (6 baselines × 4 arriving tasks = 24 jobs)
#
# Submit:
#   sbatch scripts/fedmma_array.slurm
#
# Job index mapping (SLURM_ARRAY_TASK_ID 0–23):
#   rows  0– 3 : fedmma   × {seg3d, det3d, pan2d, det2d}
#   rows  4– 7 : fedavg   × {seg3d, det3d, pan2d, det2d}
#   rows  8–11 : fedprox  × {seg3d, det3d, pan2d, det2d}
#   rows 12–15 : local    × {seg3d, det3d, pan2d, det2d}
#   rows 16–19 : sequential × {seg3d, det3d, pan2d, det2d}
#   rows 20–23 : multitask × {seg3d, det3d, pan2d, det2d}
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

#SBATCH --job-name=fedmma
#SBATCH --array=0-23
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --output=logs/fedmma_%A_%a.out
#SBATCH --error=logs/fedmma_%A_%a.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=kasrabor@buffalo.edu
#SBATCH --partition=general-compute
#SBATCH --qos=general-compute
#SBATCH --cluster=ub-hpc

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Paths
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

PROJECT=/projects/academic/alipour/kasrabor/FedMMA-fork
PT_DIR=/projects/academic/alipour/kasrabor/FedMMA-fork/waymo_pt
OUT_DIR=${PROJECT}/output/waymo
SIF=${PROJECT}/waymo_final.sif
LOGS=${PROJECT}/logs

mkdir -p "${LOGS}" "${OUT_DIR}"

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Decode job index → (baseline, arriving_task)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

BASELINES=(fedmma fedavg fedprox local sequential multitask)
TASKS=(seg3d det3d pan2d det2d)

BASELINE_IDX=$(( SLURM_ARRAY_TASK_ID / 4 ))
TASK_IDX=$(( SLURM_ARRAY_TASK_ID % 4 ))

BASELINE=${BASELINES[$BASELINE_IDX]}
ARRIVING_TASK=${TASKS[$TASK_IDX]}

echo "======================================================"
echo "Job array ID : ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
echo "Node         : $(hostname)"
echo "Start time   : $(date)"
echo "Baseline     : ${BASELINE}"
echo "Arriving task: ${ARRIVING_TASK}"
echo "PT_DIR       : ${PT_DIR}"
echo "Output dir   : ${OUT_DIR}/${BASELINE}_${ARRIVING_TASK}_seed1"
echo "======================================================"

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Common hyperparameters
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

COMMON_ARGS=(
    --arriving-task   "${ARRIVING_TASK}"
    --pt-dir          "${PT_DIR}"
    --output-dir      "${OUT_DIR}"
    --phase1-rounds   100
    --phase2-rounds   50
    --local-epochs    1
    --batch-size      8
    --frac            1.0
    --lr              1e-4
    --weight-decay    1e-2
    --prec            amp
    --eval-interval   10
    --num-clients     20
    --num-workers     8
    --pan-seq-len     5
    --eval-frac       0.1
    --seed            1
    --device          cuda
    # Model
    --adapter-dim     64
    --shared-adapter-dim 64
    --adapter-start   5
    --adapter-scale   0.1
    --num-decoder-layers 2
    --num-queries-3d  300
    --num-queries-2d  100
    --num-queries-pan 100
)

# Baseline-specific overrides
EXTRA_ARGS=()
case "${BASELINE}" in
    fedprox)
        EXTRA_ARGS+=(--mu 0.01)
        ;;
    multitask)
        # GradNorm for multi-task gradient conflict in both phases
        EXTRA_ARGS+=(--phase1-reg gradnorm --phase2-reg gradnorm --gradnorm-alpha 1.5)
        ;;
esac

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Run inside Apptainer container
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

apptainer exec \
    --nv \
    --home  "${PROJECT}" \
    "${SIF}" \
    python3 "${PROJECT}/federated_main.py" \
        --baseline "${BASELINE}" \
        "${COMMON_ARGS[@]}" \
        "${EXTRA_ARGS[@]}"

EXIT_CODE=$?

echo "======================================================"
echo "End time  : $(date)"
echo "Exit code : ${EXIT_CODE}"
echo "======================================================"

exit ${EXIT_CODE}
