#!/bin/bash

#SBATCH --job-name=slurm_hyperopt_parallel
#SBATCH --nodes=1
#SBATCH --ntasks=32
#SBATCH --time=72:00:00
#SBATCH --mem=50G
#SBATCH --partition=compute
#SBATCH --output=slurm_hyperopt_parallel_stdout_%j.out
#SBATCH --error=slurm_hyperopt_parallel_stderr_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=eskavvas@ucdavis.edu # Email to which notifications will be sent

aklog

# module load anaconda3/4.9.2
# source activate base
# source activate mb_py37
source activate mb_py385

python hyperopt_parallel.py