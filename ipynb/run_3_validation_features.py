import sys
sys.path.append('../')
import pandas as pd
import numpy as np
## my functions/classes
from mb_xai import mb_utils
import mb_xai.gut_data as gd
from sklearn.model_selection import cross_val_score, cross_validate, StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from imblearn.pipeline import make_pipeline
from sklearn.feature_selection import RFE
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import seaborn as sns

DATA_LOC = '../../../Data/microbiome_xai/'
### --- Below is for slurm! ---
# DATA_LOC = '../../Data/microbiome_xai/'

FLUX_DF_NAME = "micom_medium-fluxes-top5-736_samples.csv"
SAVE_FIG = True
LOAD_ML_PARAMS = True

N_SPLITS = 20 #10
TEST_SIZE = 0.25
SAMPLE_NUM = 200

### below is not used if LOAD_ML_PARAMS = True
C_PARAM = 100
PENALTY_PARAM = "l2"
FEAT_FILTER_TYPE = 'SelectKBest' # "RFE" #"SelectKBest"
N_FEATURES = 50

gut_data = gd.GutData()
gut_data.load_data(
    FILE_COMM_MODEL= DATA_LOC + 'reconstructions/community_50_TOP.pickle',
    FILE_GENUS_ASVS = DATA_LOC + 'agp_data/taxon_genus_asvs.csv',
    FILE_METADATA = DATA_LOC + "agp_data/metadata_biosample_filtered.csv",
    DIR_SIM_DATA = DATA_LOC + "micom-sim-data/" 
)
### Load flux dataframe
X_flux = pd.read_csv(gut_data.dir_sim_data+FLUX_DF_NAME,index_col=0)
X_flux.index = X_flux.index.astype(str)

def match_Xy_df(X_in_df, y_in_df):
    X_df, y_df = X_in_df.copy(), y_in_df.copy()
    """ Makes sure X_df and y_df have the same indices"""
    overlap = list(set(X_df.index).intersection(set(y_df.index)))
    X_df =  X_df.loc[overlap]
    y_df =  y_df.loc[overlap]
    return X_df, y_df


def get_imp_feature_series(X, n_scores, FEAT_FILTER_TYPE, phenotype, input_type):
    feat_df = pd.DataFrame()

    for est_i in range(len(n_scores["estimator"])):
        if FEAT_FILTER_TYPE == "SelectKBest":
            feat_series_df = pd.Series(n_scores["estimator"][est_i]["logisticregression"].coef_[0], index=X.columns[n_scores["estimator"][est_i][FEAT_FILTER_TYPE.lower()].get_support()],name=est_i)
        elif FEAT_FILTER_TYPE == "RFE":
            feat_series_df = pd.Series(n_scores["estimator"][est_i]["logisticregression"].coef_[0], index=X.columns[n_scores["estimator"][est_i][FEAT_FILTER_TYPE.lower()].support_],name=est_i)
        feat_df = pd.concat([feat_df, feat_series_df],axis=1)

    feat_df.fillna(0, inplace=True)
    feat_df_avg = feat_df.mean(axis=1)
    feat_df_avg.name = phenotype+"_"+input_type
    return feat_df_avg



imp_feat_flux_pheno_df, imp_feat_abundance_pheno_df = pd.DataFrame(), pd.DataFrame()
input_type_aucs_long = pd.DataFrame()
for phenotype in ["vegan", "ibs"][:]:
    gut_data = gd.GutData()
    gut_data.load_data(
        # FILE_COMM_MODEL= DATA_LOC + 'reconstructions/community_5_TOP-vegan.pickle',
        FILE_COMM_MODEL= DATA_LOC + 'reconstructions/community_50_TOP.pickle',
        FILE_GENUS_ASVS = DATA_LOC + 'agp_data/taxon_genus_asvs.csv',
        FILE_METADATA = DATA_LOC + "agp_data/metadata_biosample_filtered.csv",
        DIR_SIM_DATA = DATA_LOC + "micom-sim-data/"
    )
    gut_data.norm_abundances(filter_model=True, add_delta=True) ## Filters genus to those in model, adds small value to abundaces
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
        
    score_list = ['accuracy','balanced_accuracy','roc_auc','average_precision']

    ml_data_dict = {}
    for input_type in ["flux", "abundance"][:]:

        ### Load best ML parameters
        if LOAD_ML_PARAMS==True:
            sheet_name = phenotype+"_"+input_type
            ml_params_df_sheet = pd.read_excel(io=gut_data.dir_sim_data+"ml_params.xlsx", sheet_name=sheet_name, index_col=[0,1,2,3])
            ml_params_df_sheet.sort_values(["balanced_accuracy", "roc_auc"],ascending=False,inplace=True)
            (C_PARAM, N_FEATURES, PENALTY_PARAM, FEAT_FILTER_TYPE) = ml_params_df_sheet.index[0]
            # (C_PARAM, N_FEATURES, PENALTY_PARAM) = ml_params_df_sheet.index[0]
            print(sheet_name, C_PARAM, N_FEATURES, PENALTY_PARAM, FEAT_FILTER_TYPE)

        gut_data.logreg.C = C_PARAM
        gut_data.logreg.penalty = PENALTY_PARAM

        y_df = gut_data.y_df.copy()
        if input_type=="flux":
            X, y = match_Xy_df(X_flux.copy(), y_df)
        elif input_type=="abundance":
            X, y = match_Xy_df(gut_data.X_df.copy(), y_df)

        # X = mb_utils.filter_X_cols(X)
        # X, y = X.values, y.values

        if FEAT_FILTER_TYPE=="SelectKBest":
            feat_filter = SelectKBest(f_classif, k=N_FEATURES)
        elif FEAT_FILTER_TYPE=="RFE":
            feat_filter = RFE(estimator=gut_data.logreg, n_features_to_select=N_FEATURES)
        
        skf = StratifiedShuffleSplit(n_splits=N_SPLITS, test_size=TEST_SIZE)
        model = make_pipeline(StandardScaler(), SMOTE(), feat_filter, gut_data.logreg)
        # model = make_pipeline(StandardScaler(), RandomOverSampler(), feat_filter, gut_data.logreg)
        n_scores = cross_validate(model, X, y, scoring=score_list, cv=skf, n_jobs=-1, error_score='raise', return_estimator=True)
        score_dict = {score_id: n_scores["test_"+score_id] for score_id in score_list}

        imp_feat_df = get_imp_feature_series(X, n_scores, FEAT_FILTER_TYPE, phenotype, input_type)
        if input_type=="flux":
            imp_feat_flux_pheno_df = pd.concat([imp_feat_flux_pheno_df,imp_feat_df],axis=1)
        elif input_type=="abundance":
            imp_feat_abundance_pheno_df = pd.concat([imp_feat_abundance_pheno_df,imp_feat_df],axis=1)

        ml_data_dict.update({
            input_type: score_dict
        })

    ml_data_df = pd.DataFrame.from_dict(ml_data_dict,orient="index")
    ml_data_df.reset_index(inplace=True)
    ml_data_df_long = ml_data_df.explode("roc_auc",ignore_index=True)[["index","roc_auc"]]
    for col in ["balanced_accuracy"]:
        ml_data_df_long = pd.concat([ml_data_df_long, ml_data_df.explode(col,ignore_index=True)[col]],axis=1)

    ml_data_df_long["roc_auc"] = ml_data_df_long["roc_auc"].astype(float)
    ml_data_df_long["phenotype"] = phenotype
    input_type_aucs_long = pd.concat([input_type_aucs_long, ml_data_df_long])
        
input_type_aucs_long.to_csv(gut_data.dir_sim_data+'ml_performance.csv')
imp_feat_abundance_pheno_df.to_csv(gut_data.dir_sim_data+'imp_feat_abundance.csv')
imp_feat_flux_pheno_df.to_csv(gut_data.dir_sim_data+'imp_feat_flux.csv')


for ml_metric in ["balanced_accuracy", "roc_auc"]:
    f, ax = plt.subplots()
    ax = sns.boxplot(x="phenotype", y=ml_metric, hue="index", data=input_type_aucs_long, palette="Set2")
    ax = sns.stripplot(x="phenotype", y=ml_metric, hue="index", data=input_type_aucs_long, color=".25",  
        split=True, jitter=True,palette="Set2",linewidth=1,edgecolor='gray')
    # ax = sns.swarmplot(x="phenotype", y="roc_auc", hue="index", data=input_type_aucs_long, color=".25",  split=True, palette="Set2",edgecolor='gray')
    # ax.set_ylabel("features")
    ax.set_xlabel("trait")
    ax.set_title("LogReg %s comparison"%(ml_metric))

    if SAVE_FIG == True:
        f.savefig(gut_data.dir_sim_data+"figures/"+"LogReg_%s_comparison.svg"%(ml_metric))
        f.savefig(gut_data.dir_sim_data+"figures/"+"LogReg_%s_comparison.png"%(ml_metric))