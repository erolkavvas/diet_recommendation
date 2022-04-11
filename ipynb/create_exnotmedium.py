import sys
sys.path.append('../')
import pandas as pd
import numpy as np
## my functions/classes
from mb_xai import mb_utils
import mb_xai.gut_data as gd
from sklearn.model_selection import cross_val_score, cross_validate, StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.feature_selection import SelectKBest, f_classif
from imblearn.pipeline import make_pipeline
from sklearn.feature_selection import RFE
from imblearn.over_sampling import SMOTE

# DATA_LOC = '../../../Data/community_optimization/data/'
# DATA_LOC = '../../../Data/microbiome_xai/'
DATA_LOC = '../../../../data/ekavvas/microbiome_xai/' #aifs hpc
# FLUX_DF_NAME = "micom_medium-fluxes-top5-736_samples.csv"
### --- Below is for slurm! ---
# DATA_LOC = '../../Data/microbiome_xai/'
SAMPLE_NUM = 10000
# FLUX_DF_NAME = "micom_medium-fluxes-top50-9285_samples_fd.csv"
# FLUX_DF_NAME = "micom_medium-fluxes-top50-9285_samples_fd.csv"
N_SPLITS = 50 #10
TEST_SIZE = 0.25

def filter_X_cols(X_df, std_thresh=1e-3, verbose=False):
    """Drop features in X_df that are all 0, or the same number (or have very little std)"""
    if verbose==True:
        print(X_df.shape)
    X_df = X_df[X_df.columns[X_df.std()>std_thresh]]
    if verbose==True:
        print(X_df.shape)
    return X_df

def match_Xy_df(X_in_df, y_in_df):
    X_df, y_df = X_in_df.copy(), y_in_df.copy()
    """ Makes sure X_df and y_df have the same indices"""
    overlap = list(set(X_df.index).intersection(set(y_df.index)))
    X_df =  X_df.loc[overlap]
    y_df =  y_df.loc[overlap]
    return X_df, y_df

def drop_constant_cols(df):
    """Get rid of sklearn warnings that say there are constant columns..."""
    df = df.loc[:, (df != df.iloc[0]).any()]
    return df

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


# FLUX_DF_NAME = "micom_fluxes-top5-112_samples.csv" ## I usually take both
FLUX_DF_NAME = "micom_fluxes-top50-9285_samples_fd.csv" ## I usually take both
X_flux = pd.read_csv(gut_data.dir_sim_data+FLUX_DF_NAME,index_col=0, low_memory=False) # dtype=np.float64
X_flux.index = X_flux.index.astype(str)

exchange_notmedium = []
for col in X_flux.columns:
    if "EX_" in col and "__medium" not in col:
        # print(col)
        exchange_notmedium.append(col)

print(len(exchange_notmedium))
X_flux_exnotmedium = X_flux[exchange_notmedium].copy()
# X_flux_exnotmedium.head()

X_flux_exnotmedium.to_csv(gut_data.dir_sim_data+"micom_exnotmedium_fluxes-top50-9285_samples_fd.csv")