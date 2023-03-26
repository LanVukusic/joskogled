#!/bin/bash
#SBATCH --job-name=ris002-%j
#SBATCH --mem=16G
#SBATCH --output=log-%j.log
#SBATCH --reservation=ris2023
#SBATCH --partition=long
#SBATCH -N=1
singularity exec container.sif  python3 sailer.py
