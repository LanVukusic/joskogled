#!/bin/bash
#SBATCH --job-name=ris002-%j
#SBATCH --partition=gpu
#SBATCH --mem=32G
#SBATCH --output=abc
#SBATCH --reservation=ris2023
#SBATCH --gres=gpu:1
#SBATCH --threads-per-core=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --nodes=1

# nvidia-smi --query 
singularity exec --nv ../kontejner5.sif python3 main.py

# singularity exec container.sif  python3 main.py
