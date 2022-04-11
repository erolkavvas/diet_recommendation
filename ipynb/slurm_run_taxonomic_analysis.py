import sys
sys.path.append('../')
import os
import pandas as pd
import numpy as np
# import matplotlib.pyplot as plt
from mb_xai import mb_utils
from functools import reduce
# df = reduce(lambda df1,df2: pd.merge(df1,df2,on='id'), dfList)
# 
# aklog

# module load anaconda3/4.9.2
# source activate base
# source activate mb_py37

filt=True
filtasv_sample_thresh=3
filtasv_abund_thresh=0.0001
asv_count_threshold = 0
SAVE_OUTPUT = True

DATA_LOC = '../../../Data/microbiome_xai/'
FIG_SAVE_LOC = "../figures/"
# TABLE_SAVE_LOC = "../tables/"
TABLE_SAVE_LOC = "../../Microbiome_project/tables/"

dir_agora2 = "/mnt/d/Microbiome_project/reconstructions/AGORA2_recons/"
# dir_taxon = "../merged-data/"
dir_taxon_asv = DATA_LOC+"agp_data/taxonomy-data/"
dir_taxon_silva = DATA_LOC+"agp_data/silva_merged-data/"

# Second appproach, don't filter till AFTER joining dataframes
frames = []
for i in [4, 5, 6, 7, 8, 30, 33, 34, 35, 36, 37, 38, 39]:
# for i in [38, 39]:
    fn_taxon_silva_fd = "../../Microbiome_project/merged-data_fd_%d/metadata.tsv"%(i)
    print(fn_taxon_silva_fd)

    taxon_df = pd.read_csv(fn_taxon_silva_fd, sep='\t',header=0,index_col="id", skiprows=[1])
    taxon_df.columns = [x.replace(".R1", "") for x in taxon_df.columns]
    taxon_df_reset = taxon_df.reset_index().copy()
    frames.append(taxon_df_reset)
    del taxon_df
    del taxon_df_reset
    
# Now merge all the dataframes
taxon_df_merge = reduce(lambda df1,df2: pd.merge(df1,df2,on=["id","Taxon", "Confidence"], how="outer"), frames)
taxon_df_merge.set_index("id",inplace=True)

taxon_df_merge, asv_df_merged = mb_utils.get_silva_taxon_df(
        taxon_df_merge.fillna(0), filt=filt, filtasv_sample_thresh=filtasv_sample_thresh, filtasv_abund_thresh=filtasv_abund_thresh)

print(taxon_df_merge.shape)
print(len(set(taxon_df_merge.columns)))
print(len(taxon_df_merge.index.unique()))

print(asv_df_merged.shape)
print(len(set(asv_df_merged.columns)))
print(len(asv_df_merged.index.unique()))
del frames

print("...generating species, genus-species, and genus counts dataframes...")
genus_species_df = mb_utils.group_asvs(
    taxon_df_merge, asv_df_merged, grp_id="species", asv_count_threshold=asv_count_threshold)
if SAVE_OUTPUT==True:
    genus_species_df.to_csv(TABLE_SAVE_LOC + "SILVA_species_counts_fd.csv")
    
genus_species_df = mb_utils.group_asvs(
    taxon_df_merge, asv_df_merged, grp_id="genus_species", asv_count_threshold=asv_count_threshold)
if SAVE_OUTPUT==True:
    genus_species_df.to_csv(TABLE_SAVE_LOC + "SILVA_genus-species_counts_fd.csv")
    
genus_species_df = mb_utils.group_asvs(
    taxon_df_merge, asv_df_merged, grp_id="genus", asv_count_threshold=asv_count_threshold)
if SAVE_OUTPUT==True:
    genus_species_df.to_csv(TABLE_SAVE_LOC + "SILVA_genus_counts_fd.csv")