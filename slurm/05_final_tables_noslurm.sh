#!/bin/bash

# aklog
# klist

# export LC_ALL=C.UTF-8
# export LANG=C.UTF-8

# module load anaconda3/4.8.3
# module load qiime2/2020.8
# source activate qiime2-2020.8

# export TMPDIR=/share/taglab/Erol/Microbiome_project/tmpdir

for name in 13 17 18 22 23 24 25
do
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
done


## Test with batches 4 and 5. 2nd function tabulate takes a while...
# qiime feature-table transpose \
#   --i-table ./dada2_table_fd_11.qza \
#   --o-transposed-feature-table ./transposed-table_fd_11.qza

# qiime metadata tabulate \
#   --m-input-file ./silva_taxonomy_fd_11.qza \
#   --m-input-file ./transposed-table_fd_11.qza \
#   --o-visualization ./merged-data_fd_11.qzv

# qiime tools export \
#   --input-path ./merged-data_fd_11.qzv \
#   --output-path ./merged-data_fd_11
### --         Combine taxonomy table with ASV frequencies using the transposed-table.qza and silva taxonomy.         --
# qiime metadata tabulate \
#   --m-input-file silva_taxonomy.qza \
#   --m-input-file transposed-table.qza \
#   --o-visualization silva_merged-data.qzv

# qiime tools export \
#   --input-path silva_merged-data.qzv \
#   --output-path silva_merged-data