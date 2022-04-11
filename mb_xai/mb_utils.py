import os
import cobra
from os.path import join
import pandas as pd
import numpy as np
from collections import Counter
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt
import seaborn as sns

# ML packages
from sklearn.model_selection import StratifiedShuffleSplit, RepeatedStratifiedKFold
from sklearn.model_selection import permutation_test_score
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_recall_curve, roc_curve, auc, accuracy_score, matthews_corrcoef
from imblearn.over_sampling import SMOTE, ADASYN
from sklearn.linear_model import Lasso
from scipy import stats
from sklearn.feature_selection import SelectKBest, f_classif

from sklearn.metrics import average_precision_score

# %matplotlib inline

def get_genera_flux_corr(medium_react, gut_data, X_flux_notmedium,pheno="vegan",SAMPLE_NUM=10000,scale=True):
    "Performs correlation between genus-specific fluxes and specified phenotype"
    # medium_react = "EX_sprm_m__medium"#"EX_ind3ppa_m"# "EX_tma_m__medium"#"EX_fuc_L_m__medium"
    metab_id = medium_react.replace("EX_","").replace("__medium","")
    react_ids = [x.id for x in gut_data.com_model.metabolites.get_by_id(metab_id).reactions if "medium exchange" not in x.name]
    # print(X_flux_notmedium[react_ids].mean().sort_values())
    
    if pheno=="vegan":
        gut_data.set_vegan_df(sample_num=SAMPLE_NUM)
    elif pheno=="ibs":
        gut_data.set_ibs_df(sample_num=SAMPLE_NUM, add_other_diagnosis=False)
    elif pheno=="t2d":
        gut_data.set_t2d_df(sample_num=SAMPLE_NUM, add_other_diagnosis=False)
    elif pheno=="ibd":
        gut_data.set_ibd_df(sample_num=SAMPLE_NUM, add_other_diagnosis=False)
    y_df = gut_data.y_df.copy()
    X, y = match_Xy_df(X_flux_notmedium, y_df)
    
    SCALE_TYPE=StandardScaler()
    X = X[react_ids].copy()
    if scale==True:
        X_scale = SCALE_TYPE.fit_transform(X)
        f_stat, p_val = f_classif(X_scale, y)
    else:
        f_stat, p_val = f_classif(X, y)
    #fclass_df = pd.DataFrame(f_stat, index=X.columns, columns=["f_stat"])
    fclass_pval_df = pd.Series(p_val, index=X.columns)
    fclass_pval_df.name = "pval"
    # fclass_df = pd.DataFrame([f_stat, p_val], index=X.columns, columns=["f_stat", "p_val"])
    fclass_pval_df.sort_values(inplace=True)
    return fclass_pval_df
    
        
# medium_react = "EX_sprm_m__medium"
# print_genera_flux_distribution(medium_react, gut_data, X_flux_notmedium, pheno="vegan", SAMPLE_NUM=2000)

def get_sig_genus_exchange_CORR(pheno_sigreacts_dict, X_flux_notmedium, imp_feat_flux_pheno_df,gut_data,pheno="vegan",SAMPLE_NUM=10000,scale=True, SIG_CUTOFF = 0.1):
    """Returns dataframe iwth rows exchanges and columns genus. Values describe mean fluxes
    SIG_CUTOFF describes p-value threshold for the different genus-specific fluxes of a reaction (higher p-value = more genera)
    """
    map_react2genus = {}

    for input_type, input_df in [("flux", imp_feat_flux_pheno_df)]: # "flux", imp_feat_flux_pheno_df_metab
        #f, ax = plt.subplots(1, len(input_df.columns), figsize=(15, n_feats/2)) # 5 works well for n_feats=10
        col = pheno+"_"+input_type
        key_reacts = pheno_sigreacts_dict[col].index
            
        df_pheno = pd.DataFrame()
        for medium_react in key_reacts:
            #medium_react = "EX_tma_m__medium"#"EX_fuc_L_m__medium"
            metab_id = medium_react.replace("EX_","").replace("__medium","")
            react_ids = [x.id for x in gut_data.com_model.metabolites.get_by_id(metab_id).reactions if "medium exchange" not in x.name]
            # sig_genus_exchange = get_pvalues(X_flux_notmedium[react_ids].mean().sort_values(), sig_cutoff=SIG_CUTOFF)
            sig_genus_exchange = get_genera_flux_corr(medium_react, gut_data, X_flux_notmedium,pheno=pheno,scale=scale,SAMPLE_NUM=SAMPLE_NUM)
            sig_genus_exchange = sig_genus_exchange[sig_genus_exchange<SIG_CUTOFF].sort_values()
            sig_genus_exchange = X_flux_notmedium[react_ids].mean().loc[sig_genus_exchange.index]
            if len(react_ids)<=2:
                sig_genus_exchange = X_flux_notmedium[react_ids].mean()
            sig_genus_exchange.name = medium_react
            sig_genus_exchange.index = sig_genus_exchange.index.map(lambda x: x.split("__")[1])
            
            df_pheno = pd.concat([df_pheno, sig_genus_exchange],axis=1)
                
    return df_pheno

def get_sig_genus_exchange(pheno_sigreacts_dict, X_flux_notmedium, imp_feat_flux_pheno_df,gut_data, SIG_CUTOFF = 0.1):
    """Returns dataframe iwth rows exchanges and columns genus. Values describe mean fluxes
    SIG_CUTOFF describes p-value threshold for the different genus-specific fluxes of a reaction (higher p-value = more genera)
    """
    map_react2genus = {}

    for input_type, input_df in [("flux", imp_feat_flux_pheno_df)]: # "flux", imp_feat_flux_pheno_df_metab
        #f, ax = plt.subplots(1, len(input_df.columns), figsize=(15, n_feats/2)) # 5 works well for n_feats=10
        for i, col in enumerate(input_df.columns[:1]):
            key_reacts = pheno_sigreacts_dict[col].index
            
            df_pheno = pd.DataFrame()
            for medium_react in key_reacts:
                #medium_react = "EX_tma_m__medium"#"EX_fuc_L_m__medium"
                metab_id = medium_react.replace("EX_","").replace("__medium","")
                react_ids = [x.id for x in gut_data.com_model.metabolites.get_by_id(metab_id).reactions if "medium exchange" not in x.name]
                sig_genus_exchange = get_pvalues(X_flux_notmedium[react_ids].mean().sort_values(), sig_cutoff=SIG_CUTOFF)
                sig_genus_exchange = X_flux_notmedium[react_ids].mean().loc[sig_genus_exchange.index]
                if len(react_ids)<=2:
                    sig_genus_exchange = X_flux_notmedium[react_ids].mean()
                sig_genus_exchange.name = medium_react
                sig_genus_exchange.index = sig_genus_exchange.index.map(lambda x: x.split("__")[1])
                
                df_pheno = pd.concat([df_pheno, sig_genus_exchange],axis=1)
                
    return df_pheno

def get_pvalues(series_df, sig_cutoff=0.05):
    z_scores = stats.zscore(series_df, nan_policy="omit")
    z_scores = z_scores[z_scores.notnull()]
    p_values = stats.norm.sf(abs(z_scores.values))*2 #twosided - onesided *1
    p_values_df = pd.Series(p_values, index=z_scores.index)
    p_values_df_sig = p_values_df[p_values_df<sig_cutoff].sort_values()
    return p_values_df_sig

def get_sigreacts_dict(X_flux, imp_feat_flux_pheno_df, gut_data, SAMPLE_NUM=10000, SIG_CUTOFF=0.05):
    """Get significant features from metabolite weight vector"""
    pheno_sigreacts_dict = {}
    for input_type, input_df in [("flux", imp_feat_flux_pheno_df)]: # "flux", imp_feat_flux_pheno_df_metab
        for i, col in enumerate(input_df.columns):
            trait_input = col#+"_"+input_type
            z_scores = stats.zscore(input_df[trait_input], nan_policy="omit")
            z_scores = z_scores[z_scores.notnull()]
            p_values = stats.norm.sf(abs(z_scores.values))*2 #twosided - onesided *1
            p_values_df = pd.Series(p_values, index=z_scores.index)
            p_values_df_sig = p_values_df[p_values_df<SIG_CUTOFF].sort_values()
            
            if col=="vegan":
                gut_data.set_vegan_df(sample_num=SAMPLE_NUM)
            elif col=="ibs":
                gut_data.set_ibs_df(sample_num=SAMPLE_NUM, add_other_diagnosis=False)
            elif col=="t2d":
                gut_data.set_t2d_df(sample_num=SAMPLE_NUM, add_other_diagnosis=False)
            elif col=="ibd":
                gut_data.set_ibd_df(sample_num=SAMPLE_NUM, add_other_diagnosis=False)
            
            y_df = gut_data.y_df.copy()
            if input_type=="flux":
                X, y = match_Xy_df(X_flux.copy(), y_df)
            elif input_type=="abundance":
                X, y = match_Xy_df(gut_data.X_df.copy(), y_df)

            # X_scale = SCALE_TYPE.fit_transform(X)
            feat_means_dict = {}
            for feat_id in p_values_df_sig.index:
                col_yes = X.loc[y[y==1].index][feat_id]
                col_no = X.loc[y[y==0].index][feat_id]
                feat_means_dict.update({feat_id: {"yes": col_yes.mean(), "no": col_no.mean()}})
            
            p_values_df_sig = pd.DataFrame(p_values_df_sig)
            p_values_df_sig["yes"] = p_values_df_sig.index.map(lambda x: feat_means_dict[x]["yes"])
            p_values_df_sig["no"] = p_values_df_sig.index.map(lambda x: feat_means_dict[x]["no"])
            pheno_sigreacts_dict.update({trait_input: p_values_df_sig})
            
    return pheno_sigreacts_dict

def food_lasso(y_df, A_df, SAVE_LOC, SAVE_ID_ALPHA=3,SAVE_FIG=False,normalize=True):
    """Performs compressed sensing and plots pred/actual scatters and returns food signal dfs"""
    food_signal_df = pd.DataFrame()
    lasso_alpha = 10**(-SAVE_ID_ALPHA)
    f, ax = plt.subplots(1, len(y_df.columns), figsize=(15, 5)) # 5 works well for n_feats=10
    for i, pheno_id in enumerate(y_df.columns[:]):
        print("Non-zero feats",y_df[pheno_id][y_df[pheno_id]!=0].shape)
    # for pheno_id in y_df.columns:
        lasso = Lasso(alpha=lasso_alpha, normalize=normalize, max_iter=1e5)
        lasso.fit(A_df,y_df[pheno_id])
        score_val = lasso.score(A_df,y_df[pheno_id])
        # print("score:",score_val)

        y_pred = A_df.dot(lasso.coef_)
        # plt.scatter(y_df[pheno_id], y_pred)
        ax[i].scatter(y_df[pheno_id], y_pred)
        ax[i].set_title("%s: R2=%.2f, a=%.4f"%(pheno_id, score_val,lasso_alpha))
        ax[i].set_xlabel("Actual metabolite weights")
        if i==0:
            ax[i].set_ylabel("Predicted metabolite weights")

        x_pred_signal = pd.Series(lasso.coef_, index=A_df.columns)
        # x_pred_signal.sort_values()
        x_pred_signal.name = pheno_id
        food_signal_df = pd.concat([food_signal_df, x_pred_signal],axis=1)
        
    f.tight_layout()
    if SAVE_FIG == True:
        # f.tight_layout()
        f.savefig(SAVE_LOC)
        f.savefig(SAVE_LOC)
        
    return food_signal_df


def init_flux_food_df(imp_feat_flux_pheno_df, flux_pheno_direct_df, food_matrix_df, X_flux_consumed_cols, 
                      bool_concentrations=True, bool_direct_flux=True, bool_consumption=True):
    """Returns y_df (metab weight vector) and A_df (metab vs foods)"""
    if bool_direct_flux==False:
        imp_feat_flux_pheno_df["id"] = imp_feat_flux_pheno_df.index.map(lambda x: x.replace("EX_", "").replace("_m__medium", "[e]"))
        flux_id_pheno_direct_df = imp_feat_flux_pheno_df.set_index("id")
    else:
        flux_pheno_direct_df["id"] = flux_pheno_direct_df.index.map(lambda x: x.replace("EX_", "").replace("_m__medium", "[e]"))
        flux_id_pheno_direct_df = flux_pheno_direct_df.set_index("id")
        
    if bool_consumption == True:
        shared_consumed_metabs = list(set(flux_id_pheno_direct_df.index).intersection(set(X_flux_consumed_cols)))
        flux_id_pheno_direct_df = flux_id_pheno_direct_df.loc[shared_consumed_metabs].copy()

    flux_id_pheno_direct_df.fillna(0, inplace=True)

    ## binarize the matrix so 0 describes absence of chemical and 1 describes presence
    food_bool_matrix_df = food_matrix_df.copy()
    if bool_concentrations==False:
        food_bool_matrix_df[food_bool_matrix_df>0] = 1
    food_bool_matrix_df.fillna(0, inplace=True)

    ## how many important chemicals are found in food matrix?
    metab_overlap = list(set(flux_id_pheno_direct_df.index).intersection(set(food_bool_matrix_df.index)))
    print("len(metab_overlap):",len(metab_overlap))
    
    y_df = flux_id_pheno_direct_df.loc[metab_overlap].copy()
    A_df = food_bool_matrix_df.loc[metab_overlap].copy()
    return y_df, A_df

def get_top_df(input_df, col, n_feats=10):
    """
    input_df: imp_feat_flux_pheno_df or imp_feat_abundance_pheno_df
    col: vegan_flux
    """
    top_pos_df = input_df[col].sort_values(ascending=False)[:n_feats][::-1]
    top_neg_df = input_df[col].sort_values(ascending=True)[:n_feats][::-1]
    top_df = pd.DataFrame(pd.concat([top_pos_df, top_neg_df]))
    top_df['positive'] = top_df[col] > 0
    return top_df

def reindex_metab_id2name(in_df, gut_data):
    in_df.index = in_df.index.map(lambda x: gut_data.com_model.metabolites.get_by_id(x).name)
    return in_df

def get_metab_name_df(imp_feat_flux_pheno_df, gut_data):
    """Returns dataframe metabolite ids are now names for indices
    """
    imp_feat_flux_pheno_df_metab = imp_feat_flux_pheno_df.copy()
    imp_feat_flux_pheno_df_metab.index = imp_feat_flux_pheno_df_metab.index.map(lambda x: x.replace("EX_", "").replace("_m__medium", "_m"))
    imp_feat_flux_pheno_df_metab["name"] = imp_feat_flux_pheno_df_metab.index.map(lambda x: gut_data.com_model.metabolites.get_by_id(x).name)
    imp_feat_flux_pheno_df_metab = reindex_metab_id2name(imp_feat_flux_pheno_df_metab, gut_data)
    imp_feat_flux_pheno_df_metab.drop("name",inplace=True,axis=1)
    imp_feat_flux_pheno_df_metab.sort_values("vegan_flux",ascending=False)
    imp_feat_flux_pheno_df_metab.index = imp_feat_flux_pheno_df_metab.index.map(lambda x: x[:30] if len(x) > 25 else x)
    return imp_feat_flux_pheno_df_metab

def plot_performance_table(input_type_aucs_long, SAVE_ID, NO_DUMMY=True):
    """Saves results as a clean table with mean and std of performance metrics
    Args:
        input_type_aucs_long (pd.dataframe): output from feature selection
        SAVE_ID (str): _description_
        NO_DUMMY (bool, optional): _description_. Defaults to True.
    Returns:
        performance_table
    """
    # performance_dict = {}
    performance_table = pd.DataFrame()
    # score_ids = ["balanced_accuracy", "average_precision","f1", "f1_weighted"]
    for grp_id, grp in input_type_aucs_long.groupby("phenotype"):
        for input_id, input_grp in grp.groupby("index"):
            if NO_DUMMY==True and "dummy" not in input_id:
                # print(input_grp.head())
                df = input_grp.describe().loc[["mean","std"]].T.copy()
                df = df.round(decimals=2).apply(lambda x: '±'.join(x.astype(str)),axis=1)
                df.name = grp_id+"_"+input_id
                #performance_table = pd.concat([performance_table
                performance_table = performance_table.append(df)
            
    performance_table.to_csv(SAVE_ID)
    return performance_table

# print(cobra.__version__)
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


def drop_constant_cols(df):
    """Get rid of sklearn warnings that say there are constant columns..."""
    df = df.loc[:, (df != df.iloc[0]).any()]
    return df


def drop_lowstd_cols(df,std_cutoff=1e-5):
    """Drop columns with very low standard deviation, ML does weird things with these in FBA data..."""
    df = df.loc[:, abs(df.std())>std_cutoff]
    return df


def filt_samples_lowASVs(taxon_df,asv_sum_thresh=1000):
    asv_df = taxon_df.iloc[:,2:].copy()
    asv_df = asv_df.astype(int)
    
    samples_0_asvs = asv_df.sum()[asv_df.sum()==0].index.tolist()
    print("0 asvs:",len(samples_0_asvs))

    samples_10_asvs = asv_df.sum()[asv_df.sum()<10].index.tolist()
    print("<10 asvs:",len(samples_10_asvs))
    #["10317.000001125"]

    samples_1000_asvs = asv_df.sum()[asv_df.sum()<1000].index.tolist()
    print("<1,000 asvs:",len(samples_1000_asvs))

    samples_10000_asvs = asv_df.sum()[asv_df.sum()<10000].index.tolist()
    print("<10,000 asvs:",len(samples_10000_asvs))
    
    samples_asvs = asv_df.sum()[asv_df.sum()<asv_sum_thresh].index.tolist()
    
    taxon_df.drop(samples_asvs, axis=1, inplace=True)
    print("dropping", len(samples_asvs), "samples")
    return taxon_df
    #["10317.000001125"]

def filter_asvs(X_df, sample_thresh=10,relabundance_thresh=0.0001):
    """
    Taxa that were shown to be prone to blooming in the process of transporting AGP faecal 
    samples were removed from analysis. Taxonomy was assigned using the Silva 128 rRNA 
    database using the RDP Naive Bayesian Classifier algorithm default in dada2 (v1.14.1). 
    Sequence variants that were present in fewer than 50 samples (0.27% of total samples) 
    and in lower total relative abundance than 0.01% were removed from analysis.
    """
    bloom_asvs = [
        "04195686f2b70585790ec75320de0d6f",
        "4e1758e81adcb1107024ee52c0113a6f",
        "c18afe570abfe82d2f746ecc6e291bab",
        "ece1af985b63ebccd2833e9b5f0432e3",
        "30fc8af207d17abc37394fbf3d1793f7"
    ]
    try:
        X_df.drop(bloom_asvs,axis=1, inplace=True)
    except:
        print("bloom asvs not found in df")
    X_df_binary = X_df.copy()
    X_df_binary[X_df_binary>0]=1
    filt_asvs_samplethresh = X_df_binary.T[X_df_binary.sum()<sample_thresh].index.tolist()
    filt_asvs_relabundance = X_df.T[X_df.sum()/(X_df.sum().sum())<relabundance_thresh].index.tolist()
    print("FILTERED: # of asvs < sample count:",len(filt_asvs_samplethresh),", # of asvs < percent abundance:",len(filt_asvs_relabundance))
    filt_asvs = list(set(filt_asvs_samplethresh+filt_asvs_relabundance))
    X_df.drop(filt_asvs, axis=1,inplace=True)
    return X_df

# Combine taxons that have same category
# 1. Remove samples with less than 10000 ASV reads
# 2. Plot histogram of fraction of sample ASVs classified as genus/species

def group_asvs(tax_df, asv_df, grp_id="genus_species", asv_count_threshold=10000, verbose=True):
    """
    Creates dataframe where rows are taxonomy instead of ASV. This is achieved
    but summing the ASV counts for ASV of a shared taxa.
    """
    g_s_df = pd.DataFrame()
    for g_s_index, g_s_grp in tax_df[:].groupby(grp_id):
        # print(g_s_index, len(g_s_grp.index))
        g_s_series = asv_df.loc[g_s_grp.index].sum().copy()
        g_s_series.name = g_s_index
        g_s_df = g_s_df.append(g_s_series)
    if verbose==True:
        print(g_s_df.shape)
    
    # remove samples that have less than 10,000 reads
    remove_samples = list(g_s_df.sum()[g_s_df.sum()<asv_count_threshold].index)
    g_s_df.drop(remove_samples,axis=1,inplace=True)
    if verbose==True:
        print(g_s_df.shape)
    return g_s_df

def remove_samples(tax_df, asv_count_threshold=10000):
    # remove samples that have less than 10,000 reads
    remove_samples = list(tax_df.sum()[tax_df.sum()<asv_count_threshold].index)
    tax_df.drop(remove_samples,axis=1,inplace=True)
    print(tax_df.shape)
    return tax_df

def plot_asv_frac(g_s_df, ax=None, grp_id="Genus", alpha=0.5,):
    """
    Plot histogram of total ASVs per sample classified as specific tax grp
    """
    g_s_frac = 1 - g_s_df.iloc[0,]/g_s_df.sum()
    print(np.mean(g_s_frac))
    if ax==None:
        g = g_s_frac.plot(kind="hist", alpha=0.5)
    else:
        g = g_s_frac.plot(kind="hist", ax=ax, alpha=0.5)
    g.set_title("Avg %s classification percent = %.2f"%(grp_id,np.mean(g_s_frac)))
    g.set_xlabel("Fraction of ASVs described by %s"%(grp_id))
    g.set_ylabel("American Gut Project samples")
    return g


# Get information about compartmentalization
def get_com_model_info(com_model):
    print("# reactions:",len(com_model.reactions), ", # metabolites:", len(com_model.metabolites))
    print("# compartments:", len(com_model.compartments))
    comparts = com_model.compartments.keys()
    taxa_periplasm = ["p__"+x for x in com_model.taxa if "p__"+x in comparts]
    taxa_extracellular = ["e__"+x for x in com_model.taxa if "e__"+x in comparts]
    taxa_cytosolic = ["c__"+x for x in com_model.taxa if "c__"+x in comparts]
    comparts_other = set(comparts) - set(taxa_periplasm) - set(taxa_extracellular) - set(taxa_cytosolic)
    print(len(taxa_periplasm), len(taxa_extracellular), len(taxa_cytosolic), len(comparts_other))
    print(comparts_other)


def get_silva_taxon_df(taxon_df, filt=True, filtasv_sample_thresh=3, filtasv_abund_thresh=0.0001):
    """Takes the metadata.tsv from the Silva taxonomy and returns a dataframe 
    with columns Taxon, species, and genus"""
    asv_df = taxon_df.iloc[:,2:].copy()
    asv_df = asv_df.astype(int)

    if filt==True:
        ## Filter asvs
        asv_filt_df = filter_asvs(
            asv_df.T, sample_thresh=filtasv_sample_thresh,
            relabundance_thresh=filtasv_abund_thresh)
        asv_df = asv_filt_df.T
        taxon_df = taxon_df.loc[asv_filt_df.columns]
    # print(taxon_df.shape)

    taxon_df["genus_species"] = taxon_df["Taxon"].map(lambda x: x.split("g__")[1] if "g__" in x else '')
    taxon_df["genus_species"] = taxon_df["genus_species"].replace("; s__", '')
    # Counter(taxon_df["genus_species"].values).most_common()

    taxon_df["species"] = taxon_df["Taxon"].map(lambda x: x.split("s__")[1] if "s__" in x else '')
    taxon_df["species"] = taxon_df["species"].replace('uncultured_bacterium', '')
    taxon_df["species"] = taxon_df["species"].replace('uncultured_organism', '')
    taxon_df["species"] = taxon_df["species"].replace('gut_metagenome', '')
    taxon_df["species"] = taxon_df["species"].replace('metagenome', '')
    taxon_df["species"] = taxon_df["species"].replace('human_gut', '')
    taxon_df["species"] = taxon_df["species"].replace('unidentified', '')
    taxon_df["species"] = taxon_df["species"].replace('uncultured_rumen', '')
    taxon_df["species"] = taxon_df["species"].replace('uncultured_Clostridium', '')
    taxon_df["species"] = taxon_df["species"].replace('Clostridiales_bacterium', '')
    taxon_df["species"] = taxon_df["species"].replace('uncultured_Clostridiales', '')
    # Counter(taxon_df["species"].values).most_common()

    taxon_df["genus"] = taxon_df["Taxon"].map(lambda x: x.split("g__")[1].split(";")[0].split("_")[0].lower() if "g__" in x else '')
    taxon_df["genus"] = taxon_df["genus"].map(lambda x: x.replace("[","").replace("]",""))
    taxon_df["genus"] = taxon_df["genus"].replace("uncultured", "")
    taxon_df["genus"] = taxon_df["genus"].replace("family", "")
    taxon_df["family"] = taxon_df["Taxon"].map(lambda x: x.split("f__")[1] if "f__" in x else '')
    return taxon_df, asv_df


def get_agora_strain_dicts(dir_agora2):
    """ Takes in the location of the agora2 recons and returns two dictionaries
        genus_strain_dict -> keys genus and items agora2 strain files
        species_strain_dict -> keys species and items agora2 strain files
    """
    genus_strain_dict, species_strain_dict = {}, {}
    genus_species_dict = {}
    agora_species_list = []
    for agora2_file in os.listdir(dir_agora2)[:]:
        # print(agora2_file)
        genus = agora2_file.split("_")[0].lower()
        if "uncultured" in agora2_file:
            genus = agora2_file.split("_")[1].lower()
        strain = agora2_file
        # strain = "_".join(agora2_file.split("_")[1:]).split(".mat")[0]
        
        ### The items DO have .mat appended for genus_strain
        if genus not in genus_strain_dict.keys():
            genus_strain_dict.update({genus: [strain]})
        else:
            genus_strain_dict[genus].extend([strain])
            
        agora_species_list.append("_".join(strain.split("_")[:2]).lower())
        
        ### The items DO have .mat appended for species_strain
        species_name = "_".join(strain.split("_")[:2]).lower()
        if species_name not in species_strain_dict.keys():
            species_strain_dict.update({species_name: [strain]})
        else:
            species_strain_dict[species_name].extend([strain])
        
        ### For genus_species_dict, the items have the .json appended because they they aree merged recon files
        if genus not in genus_species_dict.keys():
            genus_species_dict.update({genus: [species_name+".json"]})
        elif species_name+".json" not in genus_species_dict[genus]:
            genus_species_dict[genus].extend([species_name+".json"])
    
    return genus_strain_dict, species_strain_dict, genus_species_dict


def get_agora_taxon_df(taxon_df, genus_strain_dict, species_strain_dict):
    """ Takes in the two agora dictionaries (genus_strain_dict, species_strain_dict)
    as well as the taxon_df WITH the Silva genus and species additions
    """
    print(len(genus_strain_dict.keys()))
    # agora_species_list = list(set(agora_species_list))
    agora_species_list = species_strain_dict.keys()
    agora_species_in, agora_species_notin = [], []
    for x in list(set(taxon_df["species"].values)):
        if x.lower() in agora_species_list:
            agora_species_in.append(x.lower())
        else:
            agora_species_notin.append(x.lower())
            
    agora_in, agora_notin = [], []
    for x in list(set(taxon_df["genus"].values)):
        if x in genus_strain_dict.keys():
            agora_in.append(x)
        else:
            agora_notin.append(x)
            
    print("genus: in-",len(agora_in), ", not in-",len(agora_notin))
    print("species: in-",len(agora_species_in), ", not in-", len(agora_species_notin))

    taxon_df["agora"] = taxon_df["genus"].map(lambda x: x if x in genus_strain_dict.keys() else "")
    taxon_df["agora_species"] = taxon_df["species"].map(lambda x: x.lower() if x.lower() in agora_species_list else "")
    taxon_df["agora_genus_species"] = taxon_df.apply(lambda x: x["agora_species"].lower() if x["agora_species"]!="" else x["agora"], axis=1)
    return taxon_df, agora_notin


# ----------------------------------------
# ----------------------------------------
# Machine learning utilities
# ----------------------------------------
# ----------------------------------------

def filter_X_cols(X_df, std_thresh=1e-3, verbose=False):
    """Drop features in X_df that are all 0, or the same number (or have very little std)"""
    if verbose==True:
        print(X_df.shape)
    X_df = X_df[X_df.columns[X_df.std()>std_thresh]]
    if verbose==True:
        print(X_df.shape)
    return X_df
    
### plotting functions for ROC and PR curves
def scale_inputs(X_df, pd_bool=True):
    """Highly recommended to standardize inputs for a Neural Net"""
    if pd_bool==True:
        orig_X_df = X_df.copy()
    scaler = StandardScaler()
    scaler.fit(X_df)
    X_df = scaler.transform(X_df)
    if pd_bool==True:
        X_df = pd.DataFrame(X_df, index=orig_X_df.index, columns=orig_X_df.columns)
    return X_df


def scale_inputs_minmax(X_df, pd_bool=True):
    """Stardarize inputs according to minmax scaling"""
    if pd_bool==True:
        orig_X_df = X_df.copy()
    scaler = MinMaxScaler()
    scaler.fit(X_df)
    X_df = scaler.transform(X_df)
    if pd_bool==True:
        X_df = pd.DataFrame(X_df, index=orig_X_df.index, columns=orig_X_df.columns)
    return X_df


def get_medium_fluxes(X_df_in):
    """Returns the dataframe of only the medium fluxes. takes gut_data.fluxes as input"""
    X_df = X_df_in.copy()
    medium_cols = [x for x in X_df.columns if "__medium" in x]
    X_df = X_df[medium_cols]
    return X_df


def update_recall_precision_dict(recall, precision, recall_precision_dict):
    for i, val in enumerate(recall):
        if val in recall_precision_dict.keys():
            recall_precision_dict[val].append(precision[i])
        else:
            recall_precision_dict.update({val: [precision[i]]})
    return recall_precision_dict


def get_PR_ROC(model, skf, X, y, bool_scale_inputs=True):
    """Fits model to data and evaluates it across different data splits. 
    Then returns the averages of the split evaluations
    """
    avg_AP, avg_AUC = 0, 0
    recall_precision_dict, tpr_fpr_dict = {}, {}
    auc_list, ap_list  = [],[]

    for train_index, test_index in skf.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        if bool_scale_inputs==True:
            X_train = scale_inputs(X_train,  pd_bool=False)
            X_test = scale_inputs(X_test, pd_bool=False)
        model.fit(X_train, y_train)
        
        precision, recall, thresholds = precision_recall_curve(y_test, model[1].predict_proba(X_test)[:,1])
        recall_precision_dict = update_recall_precision_dict(recall, precision, recall_precision_dict)
        AP = average_precision_score(y_test, model[1].predict_proba(X_test)[:,1])

        fpr, tpr, _ = roc_curve(y_test, model[1].predict_proba(X_test)[:,1])
        roc_auc = auc(fpr, tpr)
        auc_list.append(roc_auc)
        tpr_fpr_dict = update_recall_precision_dict(tpr, fpr, tpr_fpr_dict)
        ap_list.append(AP)
    
        avg_AP=avg_AP+AP
        avg_AUC=avg_AUC+roc_auc
        
    avg_AP = avg_AP/skf.n_splits
    avg_AUC = avg_AUC/skf.n_splits
    recall_vals, precision_vals = [],[]
    for recall_key, precision_list in recall_precision_dict.items():
        recall_vals.append(recall_key)
        precision_vals.append(np.mean(precision_list))
        
    zipped_lists = zip(recall_vals, precision_vals)
    sorted_pairs = sorted(zipped_lists)
    tuples = zip(*sorted_pairs)
    recall_vals, precision_vals = [ list(tuple) for tuple in  tuples]
        
    tpr_fpr_dict = {k: v for k, v in sorted(tpr_fpr_dict.items(), key=lambda item: item[1])}
    tpr_vals, fpr_vals = [],[]
    for tpr_key, fpr_list in tpr_fpr_dict.items():
        tpr_vals.append(tpr_key)
        fpr_vals.append(np.mean(fpr_list))
        
    zipped_lists = zip(fpr_vals, tpr_vals)
    sorted_pairs = sorted(zipped_lists)
    tuples = zip(*sorted_pairs)
    fpr_vals, tpr_vals = [ list(tuple) for tuple in  tuples]
    
    return recall_vals, precision_vals, avg_AP, tpr_vals, fpr_vals, avg_AUC, auc_list, ap_list


def numeric_metadata_col(df, col_id, encode_type="one-hot"):
    """
    Takes in a phenotype column and converts binarizes it into seperate columns (one hot encoding)
    """
    df[col_id] = df[col_id].astype(str).str.lower()
    for nan_id in ["nan", 'not applicable','not collected','unspecified']:
        df[col_id] = df[col_id].replace(nan_id, "not provided")
        
    df[col_id] = df[col_id].replace("false", "no")
    df[col_id] = df[col_id].replace("true", "yes")
    
    y = pd.get_dummies(df[col_id], prefix=col_id)
    
    return y

# diet_df = numeric_metadata_col(g_s_metadata, "diet_type", encode_type="one-hot")

def get_x_y_train_test(x, y):
    shared_index = list(set(x.index).intersection(y.index))
    x = x.loc[shared_index]
    y = y.loc[shared_index]
    return x, y
    
    
# X_df, y_df = get_x_y_train_test(asv_df.T, y_df)

def drop_notprovided(y):
    notprovided_cols = []
    for col in y.columns:
        if "not provided" in col:
            notprovided_cols.append(col)
    # print(notprovided_cols)
    y.drop(notprovided_cols, axis=1, inplace=True)
    remove_samples = list(y.sum(axis=1)[y.sum(axis=1)==0].index)
    # print("not provided samples:",len(remove_samples))
    y.drop(remove_samples, axis=0, inplace=True)
    return y

# y_df = drop_notprovided(y_df)

def train_model(X_train, X_test, y_train, y_test, model_type="lasso"):
    if model_type == "lasso":        
        alg = LogisticRegression(solver='liblinear', penalty="l2", class_weight='balanced')    
        alg.fit(X_train, y_train)
        imp = alg.coef_[0,:]
    if model_type == "rf":        
        alg = RandomForestClassifier()
        alg = RandomForestClassifier(n_estimators=512, min_samples_leaf=1, n_jobs=-1, bootstrap=True, 
                                     max_samples=0.7, class_weight='balanced')
        alg.fit(X_train, y_train)
        imp = alg.feature_importances_    
    y_pred = alg.predict_proba(X_test)[:,1]
    y_pred_class = alg.predict(X_test)
    y_true = y_test
    ##Performance Metrics:
    fpr, tpr, _ = roc_curve(y_true, y_pred)
    roc_auc = auc(fpr, tpr)
    matthew = matthews_corrcoef(y_true, y_pred_class)
    acc = accuracy_score(y_true, y_pred_class)
    return imp, roc_auc, acc, matthew


def getImportances(importances, col_names, y_col_id):
    avg_imps = np.stack(importances)
    avg_imps = pd.DataFrame(avg_imps, columns = col_names).mean(axis = 0)
    # avg_imps.sort_values(ascending=False,inplace=True)
    # avg_imps = pd.DataFrame(avg_imps,columns=["importance_"+y_col_id])
    return avg_imps


def run_trait_ml(
    y_col, y_col_id, g_s_metadata, asv_df, 
    n_splits=25, test_size=0.25, ml_type = "rf"):
    
    y_df = numeric_metadata_col(g_s_metadata, y_col, encode_type="one-hot")
    y_df = drop_notprovided(y_df)
    X_df, y_df = get_x_y_train_test(asv_df.T, y_df)
    y_df = y_df[y_col_id]
    X_df = np.log(X_df + 1.0)
    
    sss = StratifiedShuffleSplit(n_splits=n_splits, test_size=test_size)
    # sss = RepeatedStratifiedKFold(n_splits=3, n_repeats=3) #75/25 training/test split for each iteration
    importances, aucs, accs, matthews = [], [], [], []

    for train_index, test_index in sss.split(X_df, y_df):
        # print("TRAIN:", train_index, "TEST:", test_index)
        X_train, X_test = X_df.iloc[train_index], X_df.iloc[test_index]
        y_train, y_test = y_df.iloc[train_index], y_df.iloc[test_index]
        # print(Counter(y_train), Counter(y_test))

        scaler = StandardScaler().fit(X_train)
        X_train = pd.DataFrame(scaler.transform(X_train), index=X_train.index, columns=X_train.columns)
        X_test = pd.DataFrame(scaler.transform(X_test), index=X_test.index, columns=X_test.columns)

        imp, roc_auc, acc, matthew = train_model(X_train, X_test, y_train, y_test, model_type=ml_type)
        importances.append(imp)
        aucs.append(roc_auc)
        accs.append(acc)
        matthews.append(matthew)

    avg_importances = getImportances(importances, X_df.columns, y_col_id)
    return avg_importances, aucs, accs, matthews


# ----------------------------------------
# ----------------------------------------
# Assessing algorithm learning
# ----------------------------------------
# ----------------------------------------
# def assess_recommendation(data):
#     com_model = data[0][0]
#     genus_ASV = data[0][1]
#     sample_list = data[0][3]
#     coef_params = data[0][4]
#     y_df = data[0][2]
#     coef_norm_type = data[1]["coef_norm_type"]
#     alpha = data[1]["alpha"]
    
#     compute_results_list = []
#     overlap_norm_df, all_norm_df = normalize_asv_abundances(genus_ASV, com_model)

#     obj_var_list = [x.forward_variable for x in com_model.exchanges]
#     init_obj_var_dict = dict(zip(obj_var_list, coef_params))

    
#     compute_results_ret = run_micom_samples_parallel(
#             com_model, genus_ASV, sample_list, init_obj_var_dict,
#             return_fluxes=False, 
#             pfba_bool=False,
#             run_obj_params=True, 
#             processes=cpu_count(), 
#             tradeoff=0.3, 
#             atol=None) # 1e-6

#     y_test_series = y_df
#     y_score_series = pd.Series(dict(compute_results_ret))

#     loss_roc_auc, loss_log = get_error(y_test_series, y_score_series)
#     # regularization_cost = get_regularization(pd.Series(init_obj_var_dict).values, norm_type=coef_norm_type)

#     if coef_norm_type=="l1":
#         vec_ = np.abs(coef_params)
#         regularization_cost = np.sum(vec_)
#         # vec_norm = np.linalg.norm(obj_coefs, ord=1)
#     elif coef_norm_type=="l2":
#         regularization_cost = 0.5*np.dot(coef_params.T, coef_params)

#     minimization_error = loss_log + alpha*regularization_cost

#     print("ROCAUC: %f, LogLoss: %f, Norm: %f, Error(LogLoss+L1): %f"%(
#         loss_roc_auc, loss_log, regularization_cost, minimization_error))


# assess_recommendation(data)
# def get_error_calibration(y_test_series, y_score_series, pcalib_bins=5):
#     """
#     Transforms computed optimum values to class probabilities and calculates
#     various error metrics
#     1. standard scaling them
#     2. Transforming them to probabilities by sigmoid activation
#     3. Computing log loss with sigmoid activated probabilities
#     4. 
    
#     """
#     y_score_array = y_score_series.values
#     y_test_series = y_test_series.reindex(y_score_series.index)
#     y_test_array = y_test_series.values
    
#     ## Return ROC AUC
#     fpr_, tpr_, _ = roc_curve(y_test_array, y_score_array)
#     roc_auc_ = auc(fpr_, tpr_)
#     # error = 1 - roc_auc_
    
#     ## Other Errors require normalization of vector
#     y_score_Norm, y_score_StdScale = get_scaled_vec(y_score_array)
#     y_score_sigmoid_probs = 1/(1 + np.exp(-y_score_StdScale))
#     # https://scikit-learn.org/stable/modules/generated/sklearn.metrics.log_loss.html
#     loss_cross_entropy = log_loss(
#         y_test_array, y_score_sigmoid_probs, eps=1e-15, normalize=True, sample_weight=None, labels=None)
#     # Code for calibration 
#     fraction_of_positives, mean_predicted_value = calibration_curve(
#         y_test_array, y_score_sigmoid_probs, n_bins=pcalib_bins)
    
#     return (fpr_, tpr_, roc_auc_, loss_cross_entropy, fraction_of_positives, mean_predicted_value)

# fpr, tpr, loss_roc_auc, loss_log, frac_pos, mean_pred_val = get_error_calibration(
#     y_test_series, y_score_series, pcalib_bins=20)


def plot_roc_calprobs(fpr, tpr, loss_roc_auc, mean_pred_val, frac_pos):
    fig, ax = plt.subplots(1,2, figsize=(12, 4))

    # plot ROC curves
    # plt.figure()
    lw = 2
    ax[0].plot(fpr, tpr, color='darkorange',
             lw=lw, label='ROC curve (area = %0.2f)' % loss_roc_auc)
    ax[0].plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    ax[0].set_xlim([0.0, 1.0])
    ax[0].set_ylim([0.0, 1.05])
    ax[0].set_xlabel('False Positive Rate')
    ax[0].set_ylabel('True Positive Rate')
    ax[0].set_title('Receiver operating characteristic example')
    ax[0].legend(loc="lower right")
    # ax[0].show()

    # plot calibration curves
    ax[1].set_title("Probability Calibration Curves")
    ax[1].set_ylabel("Fraction of positives")
    ax[1].set_xlabel("Mean predicted value")
    ax[1].plot(mean_pred_val, frac_pos, 's-')
    ax[1].plot([0, 1], [0, 1], '--', color='gray')
    fig.tight_layout()
    return fig, ax

# f, ax = plot_roc_calprobs(fpr, tpr, loss_roc_auc, mean_pred_val, frac_pos)


# # ----------------------------------------
# # ----------------------------------------
# # Paralellization of MICOM computations using pebble
# # ----------------------------------------
# # ----------------------------------------
# from multiprocessing import cpu_count
# from pebble import ProcessPool, ThreadPool
# from concurrent.futures import TimeoutError

# Com_Model_Global=None
# Tradeoff_Global=None
# Atol_Global=None
# Obj_Dict_Global=None
# def init_Com_Model(mod, tradeoff, atol, obj_dict):
#     global Com_Model_Global
#     global Tradeoff_Global
#     global Atol_Global
#     global Obj_Dict_Global
#     Com_Model_Global = mod
#     Tradeoff_Global = tradeoff
#     Atol_Global = atol
#     Obj_Dict_Global = obj_dict
#     del mod
#     del tradeoff
#     del atol
#     del obj_dict
    

# def compute_comm_task(abundance_series_df):
#     """Cooperative tradeoff w/o utilizing parameterized objective"""
#     # initialize model with abundance data
#     # abundance_series_df = overlap_norm_df[sample_id]
#     com_model_context = init_sample_com(abundance_series_df, Com_Model_Global)

#     # perform cooperative tradeoff
#     tradeoff_sol = com_model_context.cooperative_tradeoff(
#         min_growth=0.0,
#         fraction=Tradeoff_Global,
#         fluxes=True,
#         pfba=True,
#         atol=Atol_Global,
#         rtol=None,
#     )

#     res = get_micom_results(com_model_context, tradeoff_sol, tradeoff=Tradeoff_Global, atol=Atol_Global)
#     return res

# def compute_comm_objective_task(abundance_series_df):
#     """FBA utilizing parameterized objective"""
#     # initialize model with abundance data
#     # abundance_series_df = overlap_norm_df[sample_id]
#     com_model_context = init_sample_com(abundance_series_df, Com_Model_Global)
#     com_model_context.objective.set_linear_coefficients(Obj_Dict_Global)
    
#     # perform regular FBA with new objective
#     sol = com_model.optimize(fluxes=True)

#     res = get_micom_results(com_model_context, sol, tradeoff=Tradeoff_Global, atol=Atol_Global)
#     return res


# def run_micom_samples_parallel(
#     com_model, genus_ASV, sample_list, objective_dict, run_obj_params=False, processes=cpu_count(),tradeoff=0.3, atol=1e-6

# ):
#     """run_micom_samples_parallel is a function that computes the flux state using MICOM
    
#     Parameters
#     ----------
#     com_model : MICOM community model 
#         similar to cobrapy object
#     genus_ASV : dataframe
#         pandas dataframe with index genus and columns samples
#     sample_list : list
#         list of sample IDs that are a subset of the columns in genus_ASV
#     objective_dict : dict
#         dictionary with forward and reverse reaction variables as key and their objective coefficients as items
#     run_obj_params : Bool
#         Specifies whether to do parameterized objective (True) or cooperative tradeoff (False)
#     processes : int
#         number of processers to use for parallelization
#     tradeoff: float
#         MICOM parameter for cooperative tradeoff (community vs individual growths)
#     """
    
#     compute_results_list = []
#     overlap_norm_df, all_norm_df = normalize_asv_abundances(genus_ASV, com_model)
    
#     with ProcessPool(processes, initializer=init_Com_Model, initargs=(com_model,tradeoff,atol,objective_dict)) as pool:
#         if run_obj_params==True:
#             future = pool.map(
#                 compute_comm_objective_task, 
#                 [overlap_norm_df[x] for x in sample_list if overlap_norm_df[x].isna().values.any()==False], 
#                 timeout=100)
#             future_iterable = future.result()
#         else:
#             future = pool.map(
#                 compute_comm_task, 
#                 [overlap_norm_df[x] for x in sample_list if overlap_norm_df[x].isna().values.any()==False], 
#                 timeout=100)
#             future_iterable = future.result()
#         compute_results_list.extend(list(future_iterable))

# #     except TimeoutError as error:
# #         print("function took longer than %d seconds" % error.args[1])
# #     except Exception as error:
# #         print("function raised %s" % error)

#     pool.close()
#     pool.stop()
#     pool.join()
#     return compute_results_list


# # sample_list = genus_ASV.columns[:10]
# # compute_results_ret = run_micom_samples_parallel(
# #     com_model, genus_ASV, sample_list, None, run_obj_params=False, processes=cpu_count(), tradeoff=0.3, atol=1e-6)
# # agg_results = get_agg_results(compute_results_ret)


# # sample_list = vegan_samples+non_vegan_samples[:10]

# # obj_var_list = [x.forward_variable for x in com_model.exchanges]
# # init_obj_var_dict = dict(zip(obj_var_list, np.ones((len(obj_var_list),))/len(obj_var_list)))

# # compute_results_ret = run_micom_samples_parallel(
# #     com_model, genus_ASV, sample_list, init_obj_var_dict, run_obj_params=True, processes=cpu_count(), tradeoff=0.3, atol=1e-6)
# # agg_results = get_agg_results(compute_results_ret)

