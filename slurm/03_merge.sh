#!/bin/bash

module load anaconda3/4.8.3
module load qiime2/2020.8
source activate qiime2-2020.8

# --i-tables dada2_table_fd_28.qza \
# --i-tables dada2_table_fd_29.qza \
# --i-tables dada2_table_fd_31.qza \
# --i-tables dada2_table_fd_32.qza \

qiime feature-table merge \
 --i-tables dada2_table_fd_0.qza \
 --i-tables dada2_table_fd_1.qza \
 --i-tables dada2_table_fd_2.qza \
 --i-tables dada2_table_fd_3.qza \
 --i-tables dada2_table_fd_4.qza \
 --i-tables dada2_table_fd_5.qza \
 --i-tables dada2_table_fd_6.qza \
 --i-tables dada2_table_fd_7.qza \
 --i-tables dada2_table_fd_8.qza \
 --i-tables dada2_table_fd_9.qza \
 --i-tables dada2_table_fd_10.qza \
 --i-tables dada2_table_fd_11.qza \
 --i-tables dada2_table_fd_12.qza \
 --i-tables dada2_table_fd_13.qza \
 --i-tables dada2_table_fd_14.qza \
 --i-tables dada2_table_fd_15.qza \
 --i-tables dada2_table_fd_16.qza \
 --i-tables dada2_table_fd_17.qza \
 --i-tables dada2_table_fd_18.qza \
 --i-tables dada2_table_fd_19.qza \
 --i-tables dada2_table_fd_20.qza \
 --i-tables dada2_table_fd_21.qza \
 --i-tables dada2_table_fd_22.qza \
 --i-tables dada2_table_fd_23.qza \
 --i-tables dada2_table_fd_24.qza \
 --i-tables dada2_table_fd_25.qza \
 --i-tables dada2_table_fd_26.qza \
 --i-tables dada2_table_fd_27.qza \
 --i-tables dada2_table_fd_30.qza \
 --i-tables dada2_table_fd_33.qza \
 --i-tables dada2_table_fd_34.qza \
 --i-tables dada2_table_fd_35.qza \
 --i-tables dada2_table_fd_36.qza \
 --i-tables dada2_table_fd_37.qza \
 --i-tables dada2_table_fd_38.qza \
 --i-tables dada2_table_fd_39.qza \
 --o-merged-table dada2_table_fd_merged.qza


#  --i-data dada2_rep_set_fd_28.qza \
#  --i-data dada2_rep_set_fd_29.qza \
#  --i-data dada2_rep_set_fd_31.qza \
#  --i-data dada2_rep_set_fd_32.qza \

qiime feature-table merge-seqs \
 --i-data dada2_rep_set_fd_0.qza \
 --i-data dada2_rep_set_fd_1.qza \
 --i-data dada2_rep_set_fd_2.qza \
 --i-data dada2_rep_set_fd_3.qza \
 --i-data dada2_rep_set_fd_4.qza \
 --i-data dada2_rep_set_fd_5.qza \
 --i-data dada2_rep_set_fd_6.qza \
 --i-data dada2_rep_set_fd_7.qza \
 --i-data dada2_rep_set_fd_8.qza \
 --i-data dada2_rep_set_fd_9.qza \
 --i-data dada2_rep_set_fd_10.qza \
 --i-data dada2_rep_set_fd_11.qza \
 --i-data dada2_rep_set_fd_12.qza \
 --i-data dada2_rep_set_fd_13.qza \
 --i-data dada2_rep_set_fd_14.qza \
 --i-data dada2_rep_set_fd_15.qza \
 --i-data dada2_rep_set_fd_16.qza \
 --i-data dada2_rep_set_fd_17.qza \
 --i-data dada2_rep_set_fd_18.qza \
 --i-data dada2_rep_set_fd_19.qza \
 --i-data dada2_rep_set_fd_20.qza \
 --i-data dada2_rep_set_fd_21.qza \
 --i-data dada2_rep_set_fd_22.qza \
 --i-data dada2_rep_set_fd_23.qza \
 --i-data dada2_rep_set_fd_24.qza \
 --i-data dada2_rep_set_fd_25.qza \
 --i-data dada2_rep_set_fd_26.qza \
 --i-data dada2_rep_set_fd_27.qza \
 --i-data dada2_rep_set_fd_30.qza \
 --i-data dada2_rep_set_fd_33.qza \
 --i-data dada2_rep_set_fd_34.qza \
 --i-data dada2_rep_set_fd_35.qza \
 --i-data dada2_rep_set_fd_36.qza \
 --i-data dada2_rep_set_fd_37.qza \
 --i-data dada2_rep_set_fd_38.qza \
 --i-data dada2_rep_set_fd_39.qza \
 --o-merged-data dada2_rep_set_fd_merged.qza


# ## Taxonomic classification using Silva dataset
# qiime feature-classifier classify-sklearn \
#   --i-reads ./dada2_rep_set_fd_merged.qza \
#   --i-classifier ./silva-138-99-515-806-nb-classifier.qza \
#   --o-classification ./silva_taxonomy_fd.qza

### --         Combine taxonomy table with ASV frequencies.         --
### takes input of dada2_table.qza, silva_taxonomy.qza, dada2_rep_set.qza
### outputs folder silva_merged-data with a file inside "metadata.tsv"
# qiime feature-table transpose \
#   --i-table dada2_table_fd_merged.qza \
#   --o-transposed-feature-table transposed-table_fd_merged.qza

# qiime metadata tabulate \
#   --m-input-file silva_taxonomy_fd.qza \
#   --m-input-file transposed-table_fd_merged.qza \
#   --o-visualization silva_merged-data_fd.qzv

# qiime tools export \
#   --input-path merged-data_fd.qzv \
#   --output-path merged-data_fd