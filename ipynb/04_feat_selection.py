import sys
sys.path.append('../')
import pandas as pd
import numpy as np
## my functions/classes
from mb_xai import mb_utils
import mb_xai.gut_data as gd
import sklearn
from sklearn.model_selection import cross_val_score, cross_validate, StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.feature_selection import SelectKBest, f_classif
from imblearn.pipeline import make_pipeline
from sklearn.feature_selection import RFE
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from tqdm import tqdm

# DATA_LOC = '../../../Data/community_optimization/data/'
DATA_LOC = '../../../Data/microbiome_xai/'

# FLUX_DF_NAME = "micom_medium-fluxes-top5-736_samples.csv"
FLUX_DF_NAME = "micom_medium-fluxes-top50-9285_samples_fd.csv"
FLUX_DF_NAME_all = "micom_fluxes-top50-9285_samples_fd.csv" ## I usually take both
FLUX_DF_NAME_all_FILT = "micom_fluxes-top50-9285_samples_fd_VEGAN-2000.csv"
FILE_PARAMS = '../../../Data/microbiome_xai/micom-sim-data/ml_params_fd_PARALLEL_top5.csv'
SAVE_FIG = True
BOOL_FLUX_NOTMEDIUM = True # decides whether to use medium fluxes or genus-specific exchange fluxes
LOAD_FILTERED_NOTMEDIUM = True
SAMPLE_NUM_NOTMEDIUM = 2000
PHENO_NOTMEDIUM = "vegan"

### ---- PARAMETERS FOR ML -----
SAVE_ID = "notmedium_2000"#"PARALLEL"#"stdscale", "maxmin" # "minmax"

N_SPLITS = 50 #10
TEST_SIZE = 0.25
SAMPLE_NUM = 10000
C_PARAM = 5
PENALTY_PARAM = "l2"#"l2"
FEAT_FILTER_TYPE = 'SelectKBest' # "RFE" #"SelectKBest"
N_FEATURES = 75
VARIANCE_THRESH = 1e-4
# SCALE_TYPE = StandardScaler()# MinMaxScaler()# StandardScaler() #StandardScaler() #MinMaxScaler()
SCALE_TYPE = StandardScaler() #MinMaxScaler()
SCALE_TYPE_ ="StandardScaler"

LOAD_ML_PARAMS = False
ONLY_FOOD_METABS = False

### ---- Load gut model -----
gut_data = gd.GutData()
gut_data.load_data(
    FILE_COMM_MODEL= DATA_LOC + 'reconstructions/community_top50_fd.pickle',
    FILE_GENUS_ASVS = DATA_LOC + 'agp_data/SILVA_genus_counts_fd.csv',
    FILE_METADATA = DATA_LOC + "agp_data/metadata_biosample_filtered.csv",
    DIR_SIM_DATA = DATA_LOC + "micom-sim-data/"  # "../data/micom-sim-data/",
)
### ---- Load flux dataframe -----

X_flux = pd.read_csv(gut_data.dir_sim_data+FLUX_DF_NAME,index_col=0, low_memory=False)
X_flux.index = X_flux.index.astype(str)
X_flux = mb_utils.drop_constant_cols(X_flux)
X_flux = mb_utils.drop_lowstd_cols(X_flux,std_cutoff=VARIANCE_THRESH)

if BOOL_FLUX_NOTMEDIUM==True:
    ## Get key columns
    if LOAD_FILTERED_NOTMEDIUM==True:
        X_flux = pd.read_csv(gut_data.dir_sim_data+FLUX_DF_NAME_all_FILT,index_col=0,low_memory=False)
        X_flux.index = X_flux.index.astype(str)
    else:
        exchange_notmedium = []
        for react in tqdm(gut_data.com_model.reactions):
            react_id = react.id
            react_name = react.name
            if "EX_" in react_id and "__medium" not in react_id and "medium exchange" not in react_name:
                # print(col)
                exchange_notmedium.append(react_id)
                
        exchange_notmedium.append("Unnamed: 0")
        exchange_notmedium = list(set(exchange_notmedium)-set(list(['EX_tDHNACOA(e)__lactobacillus'])))
        print("",len(exchange_notmedium))
        
        ## Get smaller set of samples
        phenotype = PHENO_NOTMEDIUM
        gut_data.norm_abundances(filter_model=False, add_delta=True) ## Filters genus to those in model, adds small value to abundaces
        gut_data.X_df = gut_data.asv_df.T.copy()
        gut_data.sample_list = gut_data.X_df.index.to_list()
        if phenotype=="vegan":
            gut_data.set_vegan_df(sample_num=SAMPLE_NUM_NOTMEDIUM)
        elif phenotype=="ibs":
            gut_data.set_ibs_df(sample_num=SAMPLE_NUM_NOTMEDIUM, add_other_diagnosis=False)
        elif phenotype=="t2d":
            gut_data.set_t2d_df(sample_num=SAMPLE_NUM_NOTMEDIUM, add_other_diagnosis=False)
        elif phenotype=="ibd":
            gut_data.set_ibd_df(sample_num=SAMPLE_NUM_NOTMEDIUM, add_other_diagnosis=False)
            
        index_subset = []
        for i in gut_data.y_df.index:
            index_subset.append(X_flux.index.get_loc(i)+1)

        index_subset_skip = set(list(range(len(X_flux.index))))-set(index_subset) - set([0])
        print("len(index_subset):",len(index_subset), ", len(index_subset_skip):",len(index_subset_skip))
        ## Takes 49.8s for 50 samples and 15235 columns
        ## Takes 51.3s for 100 samples and 15235 columns
        ## Takes 59.7s for 1000 samples and 15235 columns
        ## Takes 1min 5.5s for 2000 samples and 15235 columns
        ## DOESNT work for 5000 samples and 15235 columns
        X_flux = pd.read_csv(gut_data.dir_sim_data+FLUX_DF_NAME_all,index_col=0, skiprows=index_subset_skip, usecols=exchange_notmedium, low_memory=False) # dtype=np.float64
        X_flux.index = X_flux.index.astype(str)
        
    X_flux = mb_utils.drop_constant_cols(X_flux)
    X_flux = mb_utils.drop_lowstd_cols(X_flux,std_cutoff=VARIANCE_THRESH)
    print("X_flux.shape:",X_flux.shape)

# load food matrix
food_matrix_df = pd.read_csv(DATA_LOC+'tables/food_matrix_df_true.csv',index_col=0)
food_matrix_df.index = food_matrix_df.index.map(lambda x: "EX_"+x.replace("[e]", "_m__medium"))

### ---- ML algorithm ----
ml_params_df_sheet = pd.read_csv(FILE_PARAMS, index_col=0)

if ONLY_FOOD_METABS==True: ## CAREFUL... IF SET THEN X_flux is changed!!!
    metab_overlap = list(set(X_flux.columns).intersection(set(food_matrix_df.index)))
    print(len(metab_overlap))
    X_flux = X_flux[metab_overlap]
    SAVE_ID = "foodmetabs"

imp_feat_flux_pheno_df, imp_feat_abundance_pheno_df = pd.DataFrame(), pd.DataFrame()
flux_pheno_direct_df = pd.DataFrame()
input_type_aucs_long = pd.DataFrame()

for phenotype in ["vegan", "ibs","t2d", "ibd"]:
    gut_data.norm_abundances(filter_model=False, add_delta=True) ## Filters genus to those in model, adds small value to abundaces
    gut_data.X_df = gut_data.asv_df.T.copy()
    #print(gut_data.X_df.shape)
    #print(gut_data.X_df.iloc[:5,:3])
    gut_data.sample_list = gut_data.X_df.index.to_list()

    if phenotype=="vegan":
        gut_data.set_vegan_df(sample_num=SAMPLE_NUM)
    elif phenotype=="ibs":
        gut_data.set_ibs_df(sample_num=SAMPLE_NUM, add_other_diagnosis=False)
    elif phenotype=="t2d":
        gut_data.set_t2d_df(sample_num=SAMPLE_NUM, add_other_diagnosis=False)
    elif phenotype=="ibd":
        gut_data.set_ibd_df(sample_num=SAMPLE_NUM, add_other_diagnosis=False)
    # print("gut_data.X_df.shape after set_pheno_df:",gut_data.X_df.shape)
    score_list = ['accuracy','balanced_accuracy','roc_auc','average_precision', 'f1', 'f1_weighted']

    ml_data_dict = {}
    for input_type in ["flux", "abundance"][:]:

        ### Load best ML parameters
        if LOAD_ML_PARAMS==True:
            sheet_name = phenotype+"_"+input_type
            pheno_input_df = ml_params_df_sheet[ml_params_df_sheet["pheno_input"]==sheet_name].copy()
            best_params = pheno_input_df.sort_values("roc_auc",ascending=False).iloc[0]
            C_PARAM = best_params["reg_val"]
            N_FEATURES = best_params["n_features"]
            PENALTY_PARAM = best_params["reg_type"] 
            FEAT_FILTER_TYPE = 'SelectKBest'
            SCALE_TYPE_ = best_params["norm_type"] 
            
            if sheet_name == "t2d_flux":
                FEAT_FILTER_TYPE = 'SelectKBest'
                C_PARAM, N_FEATURES, PENALTY_PARAM, SCALE_TYPE_ = 0.1, 75, "l2", "StandardScaler"
                # print("CHANGED TO...",phenotype, input_type, best_params["roc_auc"], C_PARAM, N_FEATURES, PENALTY_PARAM, FEAT_FILTER_TYPE)
            if sheet_name == "vegan_flux":
                FEAT_FILTER_TYPE = 'SelectKBest'
                # C_PARAM, N_FEATURES, PENALTY_PARAM, SCALE_TYPE_ = 10, 100, "l2", "StandardScaler"
                C_PARAM, N_FEATURES, PENALTY_PARAM, SCALE_TYPE_ = 5, 40, "l2", "StandardScaler"
                # print("CHANGED TO...",phenotype, input_type, best_params["roc_auc"], C_PARAM, N_FEATURES, PENALTY_PARAM, FEAT_FILTER_TYPE)
            if sheet_name == "ibd_flux":
                FEAT_FILTER_TYPE = 'SelectKBest'
                # C_PARAM, N_FEATURES, PENALTY_PARAM, SCALE_TYPE_ = 10, 100, "l2", "StandardScaler"
                C_PARAM, N_FEATURES, PENALTY_PARAM, SCALE_TYPE_ = 5, 40, "l2", "StandardScaler"
                # print("CHANGED TO...",phenotype, input_type, best_params["roc_auc"], C_PARAM, N_FEATURES, PENALTY_PARAM, FEAT_FILTER_TYPE)
            if sheet_name == "ibs_flux":
                FEAT_FILTER_TYPE = 'SelectKBest'
                # C_PARAM, N_FEATURES, PENALTY_PARAM, SCALE_TYPE_ = 10, 100, "l2", "StandardScaler"
                C_PARAM, N_FEATURES, PENALTY_PARAM, SCALE_TYPE_ = 5, 40, "l2", "StandardScaler"
                # print("CHANGED TO...",phenotype, input_type, best_params["roc_auc"], C_PARAM, N_FEATURES, PENALTY_PARAM, FEAT_FILTER_TYPE)
                
            print(phenotype, input_type, best_params["roc_auc"], C_PARAM, N_FEATURES, PENALTY_PARAM, FEAT_FILTER_TYPE)
        else:
            print("No optimized hyperparams:",phenotype, input_type, C_PARAM, N_FEATURES, PENALTY_PARAM, FEAT_FILTER_TYPE)
                
        gut_data.logreg.C = C_PARAM
        gut_data.logreg.penalty = PENALTY_PARAM

        y_df = gut_data.y_df.copy()
        if input_type=="flux":
            X, y = mb_utils.match_Xy_df(X_flux.copy(), y_df)
        elif input_type=="abundance":
            X, y = mb_utils.match_Xy_df(gut_data.X_df.copy(), y_df)
            
        # X = mb_utils.filter_X_cols(X)
        # X, y = X.values, y.values

        if FEAT_FILTER_TYPE=="SelectKBest":
            feat_filter = SelectKBest(f_classif, k=N_FEATURES)
        elif FEAT_FILTER_TYPE=="RFE":
            feat_filter = RFE(estimator=gut_data.logreg, n_features_to_select=N_FEATURES)
            
        if SCALE_TYPE_ =="StandardScaler":
            SCALE_TYPE=StandardScaler()
        elif SCALE_TYPE_ =="MinMaxScaler":
            SCALE_TYPE=MinMaxScaler()
        
        skf = StratifiedShuffleSplit(n_splits=N_SPLITS, test_size=TEST_SIZE)
        model = make_pipeline(SCALE_TYPE, SMOTE(), feat_filter, gut_data.logreg)
        # model = make_pipeline(SMOTE(), feat_filter, gut_data.logreg)
        # model = make_pipeline(StandardScaler(), RandomOverSampler(), feat_filter, gut_data.logreg)
        n_scores = cross_validate(model, X, y, scoring=score_list, cv=skf, n_jobs=-1, error_score='raise', return_estimator=True)
        score_dict = {score_id: n_scores["test_"+score_id] for score_id in score_list}

        imp_feat_df = mb_utils.get_imp_feature_series(X, n_scores, FEAT_FILTER_TYPE, phenotype, input_type)
        if input_type=="flux":
            imp_feat_flux_pheno_df = pd.concat([imp_feat_flux_pheno_df,imp_feat_df],axis=1)

            flux_direction_dict = {}
            flux_direction_dict[input_type+"_"+phenotype] = {}
            for rxn in imp_feat_df.index:
                flux_sign = imp_feat_df.loc[rxn]*np.sign(X[rxn].loc[y[y==1].index].mean())
                flux_direction_dict[input_type+"_"+phenotype].update({rxn: flux_sign})
            flux_pheno_direct_df = pd.concat([flux_pheno_direct_df, pd.DataFrame(flux_direction_dict)],axis=1)

        elif input_type=="abundance":
            imp_feat_abundance_pheno_df = pd.concat([imp_feat_abundance_pheno_df,imp_feat_df],axis=1)

        ml_data_dict.update({
            input_type: score_dict
        })

    ### GET DUMMY ESTIMATES
    model = sklearn.dummy.DummyClassifier(strategy='uniform')
    n_scores = cross_validate(model, X, y, scoring=score_list, cv=skf, n_jobs=-1, error_score='raise', return_estimator=True)
    score_dict = {score_id: n_scores["test_"+score_id] for score_id in score_list}
    ml_data_dict.update({
            "dummy_uniform": score_dict
    })
    
    model = sklearn.dummy.DummyClassifier(strategy='most_frequent')
    n_scores = cross_validate(model, X, y, scoring=score_list, cv=skf, n_jobs=-1, error_score='raise', return_estimator=True)
    score_dict = {score_id: n_scores["test_"+score_id] for score_id in score_list}
    ml_data_dict.update({
            "dummy_frequent": score_dict
    })
    
    ### Gather results into long dataset type
    ml_data_df = pd.DataFrame.from_dict(ml_data_dict,orient="index")
    ml_data_df.reset_index(inplace=True)
    ml_data_df_long = ml_data_df.explode("roc_auc",ignore_index=True)[["index","roc_auc"]]
    for col in ["balanced_accuracy", "average_precision","f1", "f1_weighted"]:
        ml_data_df_long = pd.concat([ml_data_df_long, ml_data_df.explode(col,ignore_index=True)[col]],axis=1)

    ml_data_df_long["roc_auc"] = ml_data_df_long["roc_auc"].astype(float)
    ml_data_df_long["phenotype"] = phenotype
    input_type_aucs_long = pd.concat([input_type_aucs_long, ml_data_df_long])
        
input_type_aucs_long.to_csv(gut_data.dir_sim_data+'ml_performance_%s_fd.csv'%(SAVE_ID))
imp_feat_abundance_pheno_df.to_csv(gut_data.dir_sim_data+'imp_feat_abundance_%s_fd.csv'%(SAVE_ID))
flux_pheno_direct_df.to_csv(gut_data.dir_sim_data+'imp_flux_direction_%s_fd.csv'%(SAVE_ID))
imp_feat_flux_pheno_df.to_csv(gut_data.dir_sim_data+'imp_feat_flux_%s_fd.csv'%(SAVE_ID))