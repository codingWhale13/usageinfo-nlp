#!/bin/bash
#SBATCH --job-name=rlp-train
#SBATCH -A demelo-student
#SBATCH --partition sorcery
#SBATCH -C GPU_MEM:40GB
#SBATCH --gpus=1
#SBATCH --time=10:00:00
#SBATCH --mem=32G
#SBATCH --mail-user=nile.kiele@student.hpi.de
#SBATCH --mail-type=ALL
#SBATCH --verbose
#SBATCH --output=slurm.out
#SBATCH --error=slurm.err

source ~/miniconda3/etc/profile.d/conda.sh
conda activate copy

/hpi/fs00/home/nils.kiele/code/bsc2022-usageinfo/training/train.py --config=...
