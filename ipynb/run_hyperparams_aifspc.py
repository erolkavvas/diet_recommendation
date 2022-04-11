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
FLUX_DF_NAME = "micom_medium-fluxes-top50-9285_samples_fd.csv"
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
### Load flux dataframe
X_flux = pd.read_csv(gut_data.dir_sim_data+FLUX_DF_NAME,index_col=0, low_memory=False)
X_flux.index = X_flux.index.astype(str)
print(X_flux.shape)
X_flux = drop_constant_cols(X_flux)
print("X_flux.shape after dropping constant columns:",X_flux.shape)

writer = pd.ExcelWriter(gut_data.dir_sim_data+'ml_params_fd_noRFE.xlsx')

for phenotype in ["vegan", "ibs", "t2d", "ibd"]: # , "t2d", "ibd"
    print(phenotype)
    # gut_data = gd.GutData()
    # gut_data.load_data(
    #     # FILE_COMM_MODEL= DATA_LOC + 'reconstructions/community_5_TOP-vegan.pickle',
    #     FILE_COMM_MODEL= DATA_LOC + 'reconstructions/community_top50_fd.pickle',
    #     FILE_GENUS_ASVS = DATA_LOC + 'agp_data/SILVA_genus_counts_fd.csv',
    #     FILE_METADATA = DATA_LOC + "agp_data/metadata_biosample_filtered.csv",
    #     DIR_SIM_DATA = DATA_LOC + "micom-sim-data/"
    # )
    gut_data.norm_abundances(filter_model=False, add_delta=True)
    # gut_data.norm_abundances(filter_model=False, add_delta=True) ## Filters genus to those in model, adds small value to abundaces
    gut_data.X_df = gut_data.asv_df.T.copy()
    gut_data.sample_list = gut_data.X_df.index.to_list()

    if phenotype=="vegan":
        gut_data.set_vegan_df(sample_num=SAMPLE_NUM)
    elif phenotype=="ibs":
        gut_data.set_ibs_df(sample_num=SAMPLE_NUM, add_other_diagnosis=False)
    elif phenotype=="t2d":
        gut_data.set_t2d_df(sample_num=SAMPLE_NUM, add_other_diagnosis=False)
    elif phenotype=="ibd":
        gut_data.set_ibd_df(sample_num=SAMPLE_NUM, add_other_diagnosis=False)

    
    for input_type in ["flux", "abundance"]:

        y_df = gut_data.y_df.copy()
        if input_type=="flux":
            X, y = match_Xy_df(X_flux.copy(), y_df)
        elif input_type=="abundance":
            X, y = match_Xy_df(gut_data.X_df.copy(), y_df)

        # X = mb_utils.filter_X_cols(X)
        X, y = X.values, y.values

        score_list = ['accuracy','balanced_accuracy','roc_auc','average_precision', 'f1']

        param_score_dict = {}
        for c_param in [1e-3, 1e-2, 1e-1, 1, 5, 1e1, 1e2, 1e3]:
            for n_features in [5, 10, 15, 20, 35, 50, 75, 100, 150, 200]:
                for penalty_param in ["l1", "l2"]:

                    gut_data.logreg.C = c_param
                    gut_data.logreg.penalty = penalty_param

                    # for feat_filter in [SelectKBest(f_classif, k=n_features), RFE(estimator=gut_data.logreg, n_features_to_select=n_features)]:
                    for feat_filter in [SelectKBest(f_classif, k=n_features)]:

                        for scale_type in [StandardScaler(), MinMaxScaler()]:

                            # feat_filter = RFE(estimator=gut_data.logreg, n_features_to_select=n_features)
                            # feat_filter = SelectKBest(f_classif, k=n_features)

                            skf = StratifiedShuffleSplit(n_splits=N_SPLITS, test_size=TEST_SIZE)
                            # model = make_pipeline(StandardScaler(), SMOTE(), feat_filter, gut_data.logreg)
                            model = make_pipeline(scale_type, SMOTE(), feat_filter, gut_data.logreg)
                            # model = make_pipeline(StandardScaler(), RandomOverSampler(), feat_filter, gut_data.logreg)
                            n_scores = cross_validate(model, X, y, scoring=score_list, cv=skf, n_jobs=-1, error_score='raise')
                            score_dict = {score_id: n_scores["test_"+score_id].mean() for score_id in score_list}
                            param_score_dict.update({
                                (c_param, n_features, penalty_param, feat_filter.__class__.__name__, scale_type.__class__.__name__): score_dict})

        score_df = pd.DataFrame(param_score_dict).T
        # write dataframe to excel sheet i.e., vegan_flux, vegan_abundance
        score_df.to_excel(writer, phenotype+"_"+input_type)


# save the excel file
writer.save()