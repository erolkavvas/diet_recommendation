#!/bin/bash

#SBATCH --job-name=run_2_ml_params_aifshpc
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=72:00:00
#SBATCH --mem=50G
#SBATCH --partition=compute
#SBATCH --output=run_2_ml_params_aifshpc_stdout_%j.out
#SBATCH --error=run_2_ml_params_aifshpc_stderr_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=eskavvas@ucdavis.edu # Email to which notifications will be sent

aklog

# module load anaconda3/4.9.2
# source activate base
# source activate mb_py37
source activate mb_py385

python run_hyperparams_aifspc_multiexcel.py