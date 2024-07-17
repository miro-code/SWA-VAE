#! /bin/bash
#SBATCH --nodes=1
#SBATCH --clusters=htc
#SBATCH --job-name=swa-vae
#SBATCH --time=11:59:00
#SBATCH --gres=gpu:1
#SBATCH --partition=short
#SBATCH --account=engs-pnpl
#SBATCH --output=results/slurm_out/%j.out

module load Anaconda3/2022.10
conda activate /data/coml-oxmedis/trin4076/torch_env

export WANDB_CACHE_DIR=$DATA/wandb_cache

python vae_swa_demo.py