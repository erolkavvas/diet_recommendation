#!/bin/bash

#SBATCH --job-name=denoise_sequential
#SBATCH --nodes=1
#SBATCH --ntasks=64
#SBATCH --time=168:00:00
#SBATCH --mem=50G
#SBATCH --partition=production
#SBATCH --output=stdout_%j.out
#SBATCH --error=stderr_%j.err
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

# echo "My SLURM_ARRAY_TASK_ID: " $SLURM_ARRAY_TASK_ID
#name=`sed "${SLURM_ARRAY_TASK_ID}q;d" samples.txt`
# echo $name

# for name in 0 1 3 6 7 8 9 10 12 13 14
for name in 1 2 3
	echo $name
do
  qiime dada2 denoise-single \
    --i-demultiplexed-seqs ./demux_seqs_fd_$name.qza \
    --p-trunc-len 150 \
    --p-n-threads 0 \
    --o-table ./dada2_table_fd_$name.qza \
    --o-representative-sequences ./dada2_rep_set_fd_$name.qza \
    --o-denoising-stats ./dada2_stats_fd_$name.qza
done

echo denoising done!
