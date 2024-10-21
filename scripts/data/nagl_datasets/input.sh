#!/bin/bash

#SBATCH -A DCCADD
#SBATCH --mail-type=ALL
#SBATCH --export=ALL
#SBATCH -c 8
#SBATCH --mem=32gb

export PYTHONUNBUFFERED=FALSE

python /mnt/storage/nobackup/nca121/nagl-mbis/scripts/data/nagl_datasets/append_parquets.py > out.txt
