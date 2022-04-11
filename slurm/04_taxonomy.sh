#!/bin/bash

#SBATCH --job-name=taxonomy
#SBATCH --nodes=1
#SBATCH --ntasks=64
#SBATCH --time=168:00:00
#SBATCH --mem=50G
#SBATCH --partition=production
#SBATCH --output=stdout_taxonomy_%j.out
#SBATCH --error=stderr_taxonomy_%j.err
##SBATCH --array=1-15 # this specifies the number of jobs to run in array.. corresponds to the files in my directory
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

qiime feature-classifier classify-sklearn \
	--i-reads ./dada2_rep_set_fd_merged.qza \
	--i-classifier ./silva-138-99-515-806-nb-classifier-v2020-8.qza \
	--o-classification ./silva_taxonomy_fd_merged.qza
