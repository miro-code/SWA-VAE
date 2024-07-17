#! /bin/bash
#SBATCH --nodes=1
#SBATCH --job-name=build-env
#SBATCH --time=12:00:00
#SBATCH --partition=short
#SBATCH --account=engs-pnpl
#SBATCH --output=results/slurm_out/%j.out

module load Anaconda3/2022.10
export CONPREFIX=$DATA/torch_env
conda create --prefix $CONPREFIX 
conda activate $CONPREFIX
conda install pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 pytorch-cuda=11.7 -c pytorch -c nvidia 