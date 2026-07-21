#!/bin/bash
#SBATCH --job-name=psdiff-citibike-es
#SBATCH --account=lederer_gpu
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --time=12:00:00
#SBATCH --export=NONE
#SBATCH --array=0-2
#SBATCH --output=/beegfs/u/bbd2131/point-set-diffusion/logs/slurm-%A_%a.out

source /sw/batch/init.sh
module switch env env/2025Q1-gcc-openmpi
source "$USW/envs/ps-diff/bin/activate"
cd "$USW/point-set-diffusion"

#persistente Verzeichnisse
export WANDB_DIR="$BEEGFS/point-set-diffusion/logs"
RUN_DIR="$BEEGFS/point-set-diffusion/outputs/${SLURM_JOB_NAME}/${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
mkdir -p "$WANDB_DIR" "$RUN_DIR"


# Automatisch prüfen: hat diese Compute-Node Internet zu WandB?
if python3 -c "
import socket
socket.setdefaulttimeout(5)
socket.socket(socket.AF_INET, socket.SOCK_STREAM).connect(('api.wandb.ai', 443))
" 2>/dev/null; then
    export WANDB_MODE=online
    echo "WandB: online (Internet erkannt)"
else
    export WANDB_MODE=offline
    echo "WandB: offline (kein Internet auf dieser Node) - später mit 'wandb sync' hochladen"
fi

SEEDS=(2193852 1080243 7601429)
SEED=${SEEDS[$SLURM_ARRAY_TASK_ID]}

python train.py --config-name stpp_train \
    datamodule.name=citibike \
    model_params.noise_process=thomas \
    seed="$SEED" \
    hydra.run.dir="$RUN_DIR"
