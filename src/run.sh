#!/bin/bash
#SBATCH --job-name=ris002-%j
#SBATCH --partition=gpu
#SBATCH --mem=32G
#SBATCH --output=log-%j.log
#SBATCH --reservation=ris2023
#SBATCH --gres=gpu:1
#SBATCH --threads-per-core=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --nodes=1

singularity exec container.sif  python3 main.py
