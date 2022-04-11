import sys
sys.path.append('../')
import pandas as pd
import numpy as np
## my functions/classes
from mb_xai import mb_utils
import mb_xai.gut_data as gd

def get_genus_exchange_df(X_df):
    """returns columns that are exchange reactions of specific genera"""
    medium_cols = [x for x in X_df.columns if "__medium" in x]
    genus_ex_fluxes = []
    for med_flux in medium_cols:
        ex_flux_id = med_flux.split("_m")[0]+"(e)"
        for col in X_df.columns:
            if ex_flux_id in col:
                genus_ex_fluxes.append(col)
    genus_ex_df = X_df[genus_ex_fluxes].copy()
    return genus_ex_df

def get_genus_exchange_longform(X_df):
    """converts genera exchange reaction df to long form"""
    genus_exchange_df_melt = X_df.copy()
    columns_vals = list(genus_exchange_df_melt.columns)
    genus_exchange_df_melt = genus_exchange_df_melt.reset_index()
    genus_exchange_df_melt = pd.melt(genus_exchange_df_melt, id_vars=['index'], value_vars=columns_vals)
    # genus_exchange_df_melt[["react", "genus"]] = genus_exchange_df_melt["react"].map(lambda x: x.split("__"))
    genus_exchange_df_melt[["react", "genus"]] = genus_exchange_df_melt["react"].str.split("__", expand=True) # ,
    return genus_exchange_df_melt

# DATA_LOC = '../../../Data/community_optimization/data/'
# ATA_LOC = '../../../Data/microbiome_xai/'
DATA_LOC = '../../Data/microbiome_xai/'
SAMPLE_NUM = 10000


gut_data = gd.GutData()
gut_data.load_data(
    # FILE_COMM_MODEL='../data/reconstructions/community_5_TOP-vegan.pickle',
    # FILE_COMM_MODEL= DATA_LOC + 'reconstructions/community_5_TOP-vegan.pickle',
    # FILE_COMM_MODEL= DATA_LOC + 'reconstructions/community_50_TOP.pickle',
    FILE_COMM_MODEL= DATA_LOC + 'reconstructions/community_top50_fd.pickle',
    # FILE_GENUS_ASVS = "../data/agp_data/taxon_genus_asvs.csv",
    # FILE_GENUS_ASVS = DATA_LOC + 'agp_data/taxon_genus_asvs.csv',
    FILE_GENUS_ASVS = DATA_LOC + 'agp_data/SILVA_genus_counts_fd.csv',
    # FILE_METADATA = DATA_LOC + "agp_data/mcdonald_agp_metadata.txt",
    FILE_METADATA = DATA_LOC + "agp_data/metadata_biosample_filtered.csv",
    DIR_SIM_DATA = DATA_LOC + "micom-sim-data/"  # "../data/micom-sim-data/",
)
gut_data.norm_abundances(filter_model=True, add_delta=True) ## Filters genus to those in model, adds small value to abundaces
gut_data.X_df = gut_data.asv_df.T.copy()
gut_data.sample_list = gut_data.X_df.index.to_list()
# gut_data.set_ibs_df(sample_num=20, add_other_diagnosis=False)
# gut_data.set_vegan_df(sample_num=SAMPLE_NUM)
## Set vegan df changes X_df and y_df and will therefore change medium_df. Be sure to run medium df after setting samples
medium_df = pd.DataFrame(1000, columns=gut_data.com_model.medium.keys(), index = gut_data.X_df.index)
gut_data.sample_medium_dict = medium_df.T.to_dict()
gut_data.return_fluxes = True
gut_data.pfba_bool = True # otherwise optimum values will not be fluxes but intead min(sum flux)

## Takes about 7 minutes for 800 samples on home computer for 5 genus model...
## Crashes with top50 genus model ("function tool longer than 300 seconds")
gut_data.tradeoff_bool = True
gut_data.tradeoff_frac = 0.1
# gut_data.sample_list = gut_data.X_df.index[:10].to_list() # using 10 samples in this case
gut_data.run_micom_samples_parallel(gut_data.com_model, processes=gd.cpu_count(), atol=1e-6)

### ---- Save fluxes into a dataframe with ALL fluxes and one only including the medium fluxes
gut_data.fluxes.to_csv(gut_data.dir_sim_data+"micom_fluxes-top50-%d_samples_fd.csv"%(len(gut_data.sample_list)))
med_flux_df = mb_utils.get_medium_fluxes(gut_data.fluxes) # get medium fluxes
med_flux_df.to_csv(gut_data.dir_sim_data+"micom_medium-fluxes-top50-%d_samples_fd.csv"%(len(gut_data.sample_list)))

genus_exchange_df = get_genus_exchange_df(gut_data.fluxes)
genus_exchange_df_long = get_genus_exchange_longform(genus_exchange_df)
genus_exchange_df_long.to_csv(gut_data.dir_sim_data+"micom_medium-fluxes-genus-long-top50-%d_samples_fd.csv"%(len(gut_data.sample_list)))