#!/bin/bash

#SBATCH -A DCCADD
#SBATCH --mail-type=ALL
#SBATCH --export=ALL
#SBATCH -c 8
#SBATCH --mem=64gb

export PYTHONUNBUFFERED=FALSE

python /mnt/storage/nobackup/nca121/nagl-mbis/scripts/dataset/setup_labeled_data.py > out.txt
