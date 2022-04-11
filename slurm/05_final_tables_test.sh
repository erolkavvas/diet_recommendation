#!/bin/bash

#SBATCH --job-name=final_tables
#SBATCH --nodes=1
#SBATCH --ntasks=32
#SBATCH --time=8:00:00
#SBATCH --mem=20G
#SBATCH --partition=production
#SBATCH --output=stdout_final_tables_%j.out
#SBATCH --error=stderr_final_tables_%j.err
#SBATCH --array=1-12 # this specifies the number of jobs to run in array.. corresponds to the files in my directory
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
name=`sed "${SLURM_ARRAY_TASK_ID}q;d" samples_table_test.txt`

echo $name

qiime feature-table transpose \
  --i-table ./dada2_table_fd_$name.qza \
  --o-transposed-feature-table ./transposed-table_fd_$name.qza

qiime metadata tabulate \
  --m-input-file ./silva_taxonomy_fd_$name.qza \
  --m-input-file ./transposed-table_fd_$name.qza \
  --o-visualization ./merged-data_fd_$name.qzv

qiime tools export \
  --input-path ./merged-data_fd_$name.qzv \
  --output-path ./merged-data_fd_$name


## Test with batches 4 and 5. 2nd function tabulate takes a while...
qiime feature-table transpose \
  --i-table ./dada2_table_fd_11.qza \
  --o-transposed-feature-table ./transposed-table_fd_11.qza

qiime metadata tabulate \
  --m-input-file ./silva_taxonomy_fd_11.qza \
  --m-input-file ./transposed-table_fd_11.qza \
  --o-visualization ./merged-data_fd_11.qzv

qiime tools export \
  --input-path ./merged-data_fd_11.qzv \
  --output-path ./merged-data_fd_11
### --         Combine taxonomy table with ASV frequencies using the transposed-table.qza and silva taxonomy.         --
# qiime metadata tabulate \
#   --m-input-file silva_taxonomy.qza \
#   --m-input-file transposed-table.qza \
#   --o-visualization silva_merged-data.qzv

# qiime tools export \
#   --input-path silva_merged-data.qzv \
#   --output-path silva_merged-data