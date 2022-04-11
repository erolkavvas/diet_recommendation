#!/bin/bash

#SBATCH --job-name=run_1_micom_fluxes
#SBATCH --nodes=1
#SBATCH --ntasks=32
#SBATCH --time=24:00:00
#SBATCH --mem=10G
#SBATCH --partition=production
#SBATCH --output=run_1_micom_fluxes_stdout_%j.out
#SBATCH --error=run_1_micom_fluxes_stderr_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=eskavvas@ucdavis.edu # Email to which notifications will be sent

aklog

module load anaconda3/4.9.2
source activate base
source activate mb_py37

echo running run_1_micom_fluxes.py....

python run_1_micom_fluxes.py

