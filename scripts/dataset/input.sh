#!/bin/bash

#SBATCH --nodes 1
#SBATCH -c 288
#SBATCH --mem=0
#SBATCH --gres=gpu:4                      
#SBATCH --time=2-00:00:00   # Set the time limit to one day

# # source ~/micromamba/bin/activate splitting_env
source /home/mlpepper/${USER}/.bashrc
# source activate splitting_env
# export PYTHONUNBUFFERED=FALSE

# python ./setup_labeled_data.py > out.txt
# micromamba run -n splitting_env python ./setup_labeled_data.py > out.txt
micromamba run -n splitting_env python ./splitting.py > splitting.txt