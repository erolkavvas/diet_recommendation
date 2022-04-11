import cobra
import os
from os.path import join
import pandas as pd
from micom import Community
from micom import load_pickle
from micom.annotation import annotate_metabolites_from_exchanges
from micom.workflows.core import workflow, GrowthResults
from mb_xai.mb_utils import *
# parallel computation tools
from multiprocessing import cpu_count
from pebble import ProcessPool, ThreadPool
from concurrent.futures import TimeoutError
# import sys,argparse,resource,warnings

## module load anaconda3/4.9.2
## source activate base
## source activate mb_py37

# --- for getting cplex to work... ----
## python /share/taglab/Erol/CPLEX_Studio201/python/setup.py install

def get_micom_results(com, sol, tradeoff=1, atol=1e-6):
    """ Returns """
    ex_ids = [r.id for r in com.exchanges]
    rates = sol.members
    rates["taxon"] = rates.index
    rates["tradeoff"] = tradeoff
    rates["sample_id"] = com.id
    
    exs = list({r.global_id for r in com.internal_exchanges + com.exchanges})
    fluxes = sol.fluxes.loc[:, exs].copy()
    fluxes["sample_id"] = com.id
    fluxes["tolerance"] = atol
    anns = annotate_metabolites_from_exchanges(com)
    
    results = {"growth": rates, "exchanges": fluxes, "annotations": anns}
    # print(fluxes["tolerance"])
    
    DIRECTION = pd.Series(["import", "export"], index=[0, 1])

    growth = results["growth"]
    growth = growth[growth.taxon != "medium"]
    exchanges = results["exchanges"]
    exchanges["taxon"] = exchanges.index
    # print(exchanges["tolerance"])
    exchanges = exchanges.melt(
        id_vars=["taxon", "sample_id", "tolerance"], # ,"tolerance"
        var_name="reaction",
        value_name="flux",
    ).dropna(subset=["flux"])
    abundance = growth[["taxon", "sample_id", "abundance"]]
    exchanges = pd.merge(exchanges, abundance, on=["taxon", "sample_id"], how="outer")
    anns = results["annotations"].drop_duplicates()
    anns.index = anns.reaction
    exchanges["metabolite"] = anns.loc[exchanges.reaction, "metabolite"].values
    exchanges["direction"] = DIRECTION[(exchanges.flux > 0.0).astype(int)].values
    # exchanges = exchanges[exchanges.flux.abs() > exchanges.tolerance]

    return GrowthResults(growth, exchanges, anns)


def normalize_asv_abundances(asv_df, com_mod, verbose=False):
    """
    Returns normalized abundances of ASV counts
    
    asv_df: ASV counts in format genus (rows) vs samples (columns)
    com_mod: MICOM community model
    
    Output:
        genus_overlap_df: genus vs samples with fractional abundances based on only
            the ASVs for genus in the community model (sums to 1).
        asv_norm_df: genus vs samplesw w/ fractional abundances based on all ASVs in
            the sample (doesn't sum to 1 since only genus in the models are returned)
    """
    asv_df.index = asv_df.index.map(lambda x: x.lower() if type(x)==str else x)
    
    genus_overlap = list(set(asv_df.index).intersection(set(com_mod.abundances.index)))
    genus_absent_asv = list(set(com_mod.abundances.index)-set(asv_df.index))
    for genus in genus_absent_asv:
        asv_df = asv_df.append(pd.Series(0, index=asv_df.columns, name=genus))
        
    if verbose==True:
        print("# of genus in model and not filterd out of QA/QC",len(genus_overlap))
        print("# of genus not in model",len(genus_absent_asv))
    
    # The normalization here ignores the total number of ASVs in the sample! 
    # Also adds a row of 0s for any genus that was filtered in the asv processing earlier
    genus_overlap_df = asv_df.loc[genus_overlap].copy()
    for genus in genus_absent_asv:
        genus_overlap_df = genus_overlap_df.append(pd.Series(0, index=genus_overlap_df.columns, name=genus))
    genus_overlap_norm_df = genus_overlap_df/genus_overlap_df.sum()
    
    ### The normalization here accounts for the total number of ASVs (including NaN row!)
    asv_norm_df = asv_df/asv_df.sum()
    asv_norm_df = asv_norm_df.loc[genus_overlap_norm_df.index].copy()
    
    return genus_overlap_norm_df, asv_norm_df

def init_sample_com(abundance_series, com):
    """ Initializese the community model with the abundance data
    """
    com.set_abundance(abundance_series,normalize=True)
    com.id = abundance_series.name
    return com


# ----------------------------------------
# Functionality for computing prediction error
# ----------------------------------------
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import normalize
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import log_loss

def get_scaled_vec(y_score_array):
    y_data = pd.DataFrame(y_score_array)

    transformer = Normalizer(norm="l2").fit(y_data.T)
    y_score_scale_Norm = transformer.transform(y_data.T)
    y_score_scale_Norm = y_score_scale_Norm[0]

    scaler = StandardScaler() # with_mean=False, with_std=False
    scaler.fit(y_data)
    y_score_scale_StdScale = scaler.transform(y_data) # 0 mean, 1 std dev
    y_score_scale_StdScale = y_score_scale_StdScale[:, 0]
    # print(y_score_scale_StdScale.sum(), y_score_scale_StdScale.std())
    
    return y_score_scale_Norm, y_score_scale_StdScale

def get_error(y_test_series, y_score_series):
    y_score_array = y_score_series.values
    y_test_series = y_test_series.reindex(y_score_series.index)
    y_test_array = y_test_series.values
    
    ## Return ROC AUC
    fpr_, tpr_, _ = roc_curve(y_test_array, y_score_array)
    roc_auc_ = auc(fpr_, tpr_)
    # error = 1 - roc_auc_
    
    ## Other Errors require normalization of vector
    y_score_Norm, y_score_StdScale = get_scaled_vec(y_score_array)
    y_score_sigmoid_probs = 1/(1 + np.exp(-y_score_StdScale))
    # https://scikit-learn.org/stable/modules/generated/sklearn.metrics.log_loss.html
    loss_cross_entropy = log_loss(
        y_test_array, y_score_sigmoid_probs, eps=1e-15, normalize=True, sample_weight=None, labels=None)
    # Code for calibration 
    
    return (roc_auc_, loss_cross_entropy)

def get_regularization(obj_coefs, norm_type="l1"):
    # do not do numpy norm... its different
    if norm_type=="l1":
        vec_ = np.abs(coef_params)
        vec_norm = np.sum(vec_)
        # vec_norm = np.linalg.norm(obj_coefs, ord=1)
    elif norm_type=="l2":
        vec_norm = np.dot(obj_coefs.T, obj_coefs)
        # vec_norm = np.linalg.norm(obj_coefs, ord=2)
        
    return vec_norm


# ----------------------------------------
# ----------------------------------------
# Paralellization of MICOM computations using pebble
# ----------------------------------------
# ----------------------------------------
# Second iteration of parallelization
from multiprocessing import cpu_count
from pebble import ProcessPool, ThreadPool
from concurrent.futures import TimeoutError

Com_Model_Global=None
Tradeoff_Global=None
Atol_Global=None
Obj_Dict_Global=None
Return_Fluxes_Global=None
def init_Com_Model(mod, tradeoff, atol, obj_dict, return_fluxes, pfba_bool):
    global Com_Model_Global
    Com_Model_Global = mod
    del mod
    
    global Tradeoff_Global
    Tradeoff_Global = tradeoff
    del tradeoff
    
    global Atol_Global
    Atol_Global = atol
    del atol
    
    global Obj_Dict_Global
    Obj_Dict_Global = obj_dict
    del obj_dict
    
    global Return_Fluxes_Global
    Return_Fluxes_Global = return_fluxes
    del return_fluxes
    
    global pFBA_Global
    pFBA_Global = pfba_bool
    del pfba_bool
    

def compute_comm_task(abundance_series_df):
    """Cooperative tradeoff w/o utilizing parameterized objective"""
    # 1. Constrain model with abundances
    com_model_context = init_sample_com(abundance_series_df, Com_Model_Global)

    # 2. perform cooperative tradeoff
    tradeoff_sol = com_model_context.cooperative_tradeoff(
        min_growth=0.0,
        fraction=Tradeoff_Global,
        fluxes=Return_Fluxes_Global,
        pfba=pFBA_Global,
        atol=Atol_Global,
        rtol=None
    )

    res = get_micom_results(com_model_context, tradeoff_sol, tradeoff=Tradeoff_Global, atol=Atol_Global)
    return res


def compute_comm_objective_task(abundance_series_df):
    """FBA utilizing parameterized objective
    Output: 
        If fluxes=True return
           - GrowthResults(growth, exch, ann) object 
        else return
           - Tuple (sample, optimum value)
    """
    # 1. Constrain model with abundances
    com_model_context = init_sample_com(abundance_series_df, Com_Model_Global)
    
    # 2. Set objective function coefficients
    com_model_context.objective.set_linear_coefficients(Obj_Dict_Global)
    
    # 3. Perform FBA
    sol = com_model_context.optimize(
        fluxes=Return_Fluxes_Global,
        pfba=pFBA_Global,
        raise_error=False,
        atol=Atol_Global,
        rtol=None
    )
    
    if Return_Fluxes_Global==True:
        res = get_micom_results(com_model_context, sol, tradeoff=Tradeoff_Global, atol=Atol_Global)
        return res
    else:
    	# if sol.growth_rate
        return (abundance_series_df.name, sol.growth_rate)
    
    


# Parallel computation with a population of constrained MICOM models
def run_micom_samples_parallel(
    com_model, genus_ASV, sample_list, objective_dict, 
    return_fluxes=False, 
    pfba_bool=False,
    run_obj_params=False, 
    processes=cpu_count(),
    tradeoff=0.3, 
    atol=1e-6

):
    """Function that performs MICOM with abundance data and other parameters
    Parameters
    ----------
    com_model : MICOM community model 
        similar to cobrapy object
    genus_ASV : dataframe
        pandas dataframe with index genus and columns samples
    sample_list : list
        list of sample IDs that are a subset of the columns in genus_ASV
    objective_dict : dict
        dictionary with forward and reverse reaction variables as key and their objective coefficients as items
    run_obj_params : Bool
        Specifies whether to do parameterized objective (True) or cooperative tradeoff (False)
    processes : int
        number of processers to use for parallelization
    tradeoff: float
        MICOM parameter for cooperative tradeoff (community vs individual growths)
    """
    
    compute_results_list = []
    overlap_norm_df, all_norm_df = normalize_asv_abundances(genus_ASV, com_model)
    
    with ProcessPool(
        processes, 
        initializer=init_Com_Model, 
        initargs=(com_model, tradeoff, atol, objective_dict, return_fluxes, pfba_bool)
    ) as pool:
	    try:
	        ### FBA with specified objective coeeficients (cT)
	        if run_obj_params==True:
	            future = pool.map(
	                compute_comm_objective_task, 
	                [overlap_norm_df[x] for x in sample_list if overlap_norm_df[x].isna().values.any()==False], 
	                timeout=300)
	            future_iterable = future.result()
	            
	        ### Perform MICOM cooperative tradeoff
	        else:
	            future = pool.map(
	                compute_comm_task, 
	                [overlap_norm_df[x] for x in sample_list if overlap_norm_df[x].isna().values.any()==False], 
	                timeout=300)
	            future_iterable = future.result()

	        compute_results_list.extend(list(future_iterable))
	    except TimeoutError as error:
	        print("function took longer than %d seconds" % error.args[1])
	        compute_results_list = None
	    except Exception as error:
	        print("function raised %s" % error)
	        compute_results_list = None

    pool.close()
    pool.stop()
    pool.join()
    
    return compute_results_list


# sample_list = genus_ASV.columns[:10]
# compute_results_ret = run_micom_samples_parallel(
#     com_model, genus_ASV, sample_list, None, run_obj_params=False, processes=cpu_count(), tradeoff=0.3, atol=1e-6)
# agg_results = get_agg_results(compute_results_ret)

# sample_list = vegan_samples+non_vegan_samples[:10]

### --- Example 2 ---
# obj_var_list = [x.forward_variable for x in com_model.exchanges]
# init_obj_var_dict = dict(zip(obj_var_list, np.ones((len(obj_var_list),))/len(obj_var_list)))

# sample_list = genus_ASV.columns[:10]

# compute_results_ret = run_micom_samples_parallel(
#     com_model, genus_ASV, sample_list, init_obj_var_dict,
#     return_fluxes=False, 
#     pfba_bool=False,
#     run_obj_params=True, 
#     processes=cpu_count(), 
#     tradeoff=0.3, 
#     atol=None # 1e-6
# )
### -----------------