#!/bin/bash

#SBATCH --job-name=taxonomy_distr_1_3
#SBATCH --nodes=1
#SBATCH --ntasks=64
#SBATCH --time=336:00:00
#SBATCH --mem=50G
#SBATCH --partition=production
#SBATCH --output=stdout_taxonomy_distr_1_3_%j.out
#SBATCH --error=stderr_taxonomy_distr_1_3_%j.err
#SBATCH --array=1-26 # this specifies the number of jobs to run in array.. corresponds to the files in my directory
#SBATCH --mail-type=ALL
#SBATCH --mail-user=eskavvas@ucdavis.edu # Email to which notifications will be sent

aklog
klist

export LC_ALL=C.UTF-8
export LANG=C.UTF-8

module load anaconda3/4.8.3
module load qiime2/2020.8
source activate qiime2-2020.8

export TMPDIR=/share/taglab/Erol/Microbiome_project/tmpdir

echo "My SLURM_ARRAY_TASK_ID: " $SLURM_ARRAY_TASK_ID
name=`sed "${SLURM_ARRAY_TASK_ID}q;d" samples_1_3.txt`

echo $name

qiime feature-classifier classify-sklearn \
    --i-reads ./dada2_rep_set_fd_$name.qza \
    --i-classifier ./silva-138-99-515-806-nb-classifier-v2020-8.qza \
    --o-classification ./silva_taxonomy_fd_$name.qza
