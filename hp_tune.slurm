#!/bin/bash
#SBATCH --account=vjgo8416-ice-frcst
#SBATCH --qos=turing
#SBATCH --gres=gpu:1
#SBATCH --job-name=zebra_tune
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=32G
#SBATCH --gpus-per-task=1
#SBATCH --output=/bask/projects/v/vjgo8416-ice-frcst/users/nmjw6681/tune/tune_%j.out
#SBATCH --error=/bask/projects/v/vjgo8416-ice-frcst/users/nmjw6681/tune/tune_%j.err

module purge
module load baskerville
module load Python/3.11.3-GCCcore-12.3.0


source /bask/projects/v/vjgo8416-ice-frcst/users/nmjw6681/ice-station-zebra/.venv/bin/activate

cd /bask/projects/v/vjgo8416-ice-frcst/users/nmjw6681/ice-station-zebra

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

WANDB_MODE=offline zebra tune --config-name vitTest.local.yaml

echo "Tuning complete at $(date)"