import numpy as np
import pandas as pd
import os
from os import listdir, path, makedirs
from collections import Counter
from tqdm import tqdm
## Plotting stuff
import matplotlib.pyplot as plt
## from scikit-bio
from skbio.stats.composition import multiplicative_replacement
## Parallelization stuff
from multiprocessing import cpu_count
from pebble import ProcessPool, ThreadPool
from concurrent.futures import TimeoutError
## Micom stuff
from micom import load_pickle
from micom.media import minimal_medium
from micom.annotation import annotate_metabolites_from_exchanges
from micom.workflows.core import GrowthResults
from micom.util import _apply_min_growth
## Sklearn stuff
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
## mb_xai stuff
from mb_xai import mb_utils


class GutData(object):
    """ Class object holding information about Gut project dataset
        - Performs ML and statistical tests
    """
    def __init__(self):
        self.dir_sim_data = None
        self.asv_df = pd.DataFrame()
        self.asv_df_original = pd.DataFrame()
        self.com_model = None
        self.X_df = pd.DataFrame()
        self.y_df = pd.DataFrame()
        self.metadata_df = pd.DataFrame()
        self.sample_list = []
        ## computation parameters
        self.tradeoff_frac = 0.3 # Micom recommends tradeoff frac<0.5 for relevant growth rates, used when min_growth_rates_bool=True or when tradeoff_bool=True
        self.pfba_bool = True # whether to use pFBA or normal FBA
        self.tradeoff_bool = False # whether to run micom Cooperative tradeoff or normal optimization
        self.return_fluxes = False
        self.compute_results_list = [] # optimum values for each AGP sample
        self.fluxes = None # contains flux solutiosn if return_fluxes = True
        self.min_growth_rates_bool = False # whether to first constrain sample-specific models with min growth rates derived from Micom Cooperative tradeoff
        self.min_growth_df = pd.DataFrame() # holds sample-specific min-growth information
        self.sample_medium_dict = {}
        self.member_growth_rates = None
        self.X_flux = pd.DataFrame()
        self.rf = RandomForestClassifier(
            n_estimators=512, min_samples_leaf=1, n_jobs=-1, bootstrap=True, max_samples=0.7, class_weight='balanced')
        self.logreg = LogisticRegression(
            C=1e2,fit_intercept=True,intercept_scaling=True,solver='liblinear', penalty="l2", class_weight='balanced',max_iter=1000) 
        self.nn = MLPClassifier(
            alpha=1e-4, activation="logistic", hidden_layer_sizes=(5, 2), solver='lbfgs',max_iter=1000)
    
    
    def fix_com_model(self, verbose=True):
        """ Manually fixes minor issues in micom community metabolic models"""
        try:
            self.com_model.reactions.get_by_id("tDHNACOA__lactobacillus").id = "EX_tDHNACOA(e)__lactobacillus"
            self.com_model.reactions.get_by_id("EX_tDHNACOA(e)__lactobacillus").build_reaction_from_string('dhnacoa[e]__lactobacillus <=> 0.0023125206475057814 dhnacoa_m')
            if verbose==True:
                print("Fixed EX_tDHNACOA(e)")
        except:
            if verbose==True:
                print("Nothing to fix")
        # return com_model
    
    
    def norm_abundances(self, filter_model=False, add_delta=False, delta=None,drop_nan_genera_col=False, verbose=True):
        """Transform raw abundances to fraction of sample
        """
        if verbose==True:
            print("... normalizing raw ASV abundances to fractions, dropping samples with 0 total abundances")
        self.asv_df = self.asv_df/self.asv_df.sum()
        self.asv_df.dropna(axis=1, inplace=True)
        if drop_nan_genera_col==True:
            self.asv_df = self.asv_df.T
            self.asv_df.drop(np.nan, axis=1, inplace=True)
            self.asv_df = self.asv_df.T
            self.asv_df = self.asv_df/self.asv_df.sum()
            self.asv_df.dropna(axis=1, inplace=True)
        if add_delta==True:
            ### need to do this step after because cant work if all 0s for a sample
            self.add_delta_abundances(delta=delta)
            self.multiplicative_replacement=True
        if filter_model==True:
            self.filter_model_genus(verbose=verbose)
    
    
    def filter_model_genus(self,verbose=False):
        """Filter the asv_df to only the genus in the given community model
            ** Run norm_abundances before this step, otherwise fractions wont be scaled to total ASVs in sample
        """
        self.asv_df.index = self.asv_df.index.map(lambda x: x.lower() if type(x)==str else x)
        genus_overlap = list(set(self.asv_df.index).intersection(set(self.com_model.abundances.index)))
        genus_absent_asv = list(set(self.com_model.abundances.index)-set(self.asv_df.index))
        for genus in genus_absent_asv:
            self.asv_df = self.asv_df.append(pd.Series(0, index=self.asv_df.columns, name=genus))
        if verbose==True:
            print("# of genus in model and not filterd out of QA/QC: %d, # of genus not in model:%d"%(len(genus_overlap),len(genus_absent_asv)))
        self.asv_df = self.asv_df.loc[genus_overlap]
        self.asv_df = self.asv_df/self.asv_df.sum()
        
        
    def add_delta_abundances(self, delta=None):
        """ uses multiplicative_replacement from scikit-bio to make sure the values are non-zero
            - if delta=None, adds 1/N^2 to each non-zero value where N is number of features
        """
        self.asv_df = pd.DataFrame(multiplicative_replacement(self.asv_df.T,delta=delta),index=self.asv_df.T.index,columns=self.asv_df.T.columns)
        self.asv_df = self.asv_df.T
        self.multiplicative_replacement=True
    
    
    def match_Xy_df(self):
        """ Makes sure X_df and y_df have the same indices"""
        overlap = list(set(self.X_df.index).intersection(set(self.y_df.index)))
        self.X_df = self.X_df.loc[overlap]
        self.y_df = self.y_df.loc[overlap]
        self.sample_list = overlap
        
        
    def set_vegan_df(self, sample_num=5):
        """ Helper function for specifying a list of samples that include vegans 
            (will do something similar for IBS and other traits with strong signals)
        """
        col_type = "diet_type"
        y_df = mb_utils.numeric_metadata_col(self.metadata_df, col_type, encode_type="one-hot")
        y_df = mb_utils.drop_notprovided(y_df)

        vegan_samples = y_df["diet_type_vegan"][y_df["diet_type_vegan"]!=0].index.tolist()
        non_vegan_samples = y_df["diet_type_vegan"][y_df["diet_type_vegan"]==0].index.tolist()

        # sample_list = vegan_samples+non_vegan_samples[:len(vegan_samples)]
        sample_list = vegan_samples[:sample_num]+non_vegan_samples[:sample_num]
        y_df = y_df["diet_type_vegan"]
        y_df = y_df.loc[sample_list]
        self.y_df = y_df
        self.sample_list = sample_list
        self.X_df = self.asv_df.T #.loc[sample_list]
        ### match the indices of X and y
        self.match_Xy_df()

    def set_ibs_df(self, add_other_diagnosis=True, sample_num=5):
        """ Helper function for specifying a list of samples that include IBS
        """
        col_type = "ibs"
        y_df = mb_utils.numeric_metadata_col(self.metadata_df, col_type, encode_type="one-hot")
        y_df = mb_utils.drop_notprovided(y_df)

        ibs_samples = y_df["ibs_diagnosed by a medical professional (doctor, physician assistant)"][y_df["ibs_diagnosed by a medical professional (doctor, physician assistant)"]!=0].index.tolist()
        if add_other_diagnosis==True:
            ibs_samples_self = y_df["ibs_self-diagnosed"][y_df["ibs_self-diagnosed"]==1].index.tolist()
            ibs_samples_alt = y_df["ibs_diagnosed by an alternative medicine practitioner"][y_df["ibs_diagnosed by an alternative medicine practitioner"]!=0].index.tolist()
            ibs_samples.extend(ibs_samples_self)
            ibs_samples.extend(ibs_samples_alt)
        non_ibs_samples = y_df["ibs_i do not have this condition"][y_df["ibs_i do not have this condition"]==1].index.tolist()

        # sample_list = vegan_samples+non_vegan_samples[:len(vegan_samples)]
        sample_list = ibs_samples[:sample_num]+non_ibs_samples[:sample_num]
        y_df = y_df["ibs_diagnosed by a medical professional (doctor, physician assistant)"]
        y_df.loc[ibs_samples]=1
        y_df.loc[non_ibs_samples]=0
        y_df = y_df.loc[sample_list]
        self.y_df = y_df
        self.sample_list = sample_list
        self.X_df = self.asv_df.T #.loc[sample_list]
        ### match the indices of X and y
        self.match_Xy_df()


    def set_t2d_df(self, add_other_diagnosis=True, sample_num=5):
        """ Helper function for specifying a list of samples that include IBS
        """
        col_type = "diabetes"
        y_df = mb_utils.numeric_metadata_col(self.metadata_df, col_type, encode_type="one-hot")
        y_df = mb_utils.drop_notprovided(y_df)

        ibs_samples = y_df["diabetes_diagnosed by a medical professional (doctor, physician assistant)"][y_df["diabetes_diagnosed by a medical professional (doctor, physician assistant)"]!=0].index.tolist()
        if add_other_diagnosis==True:
            ibs_samples_self = y_df["diabetes_self-diagnosed"][y_df["diabetes_self-diagnosed"]==1].index.tolist()
            # ibs_samples_alt = y_df["ibs_diagnosed by an alternative medicine practitioner"][y_df["ibs_diagnosed by an alternative medicine practitioner"]!=0].index.tolist()
            ibs_samples.extend(ibs_samples_self)
            # ibs_samples.extend(ibs_samples_alt)
        non_ibs_samples = y_df["diabetes_i do not have this condition"][y_df["diabetes_i do not have this condition"]==1].index.tolist()

        # sample_list = vegan_samples+non_vegan_samples[:len(vegan_samples)]
        sample_list = ibs_samples[:sample_num]+non_ibs_samples[:sample_num]
        y_df = y_df["diabetes_diagnosed by a medical professional (doctor, physician assistant)"]
        y_df.loc[ibs_samples]=1
        y_df.loc[non_ibs_samples]=0
        y_df = y_df.loc[sample_list]
        self.y_df = y_df
        self.sample_list = sample_list
        self.X_df = self.asv_df.T #.loc[sample_list]
        ### match the indices of X and y
        self.match_Xy_df()


    def set_ibd_df(self, add_other_diagnosis=True, sample_num=5):
        """ Helper function for specifying a list of samples that include IBS
        """
        col_type = "ibd"
        y_df = mb_utils.numeric_metadata_col(self.metadata_df, col_type, encode_type="one-hot")
        y_df = mb_utils.drop_notprovided(y_df)

        ibs_samples = y_df["ibd_diagnosed by a medical professional (doctor, physician assistant)"][y_df["ibd_diagnosed by a medical professional (doctor, physician assistant)"]!=0].index.tolist()
        if add_other_diagnosis==True:
            ibs_samples_self = y_df["ibd_self-diagnosed"][y_df["ibd_self-diagnosed"]==1].index.tolist()
            # ibs_samples_alt = y_df["ibs_diagnosed by an alternative medicine practitioner"][y_df["ibs_diagnosed by an alternative medicine practitioner"]!=0].index.tolist()
            ibs_samples.extend(ibs_samples_self)
            # ibs_samples.extend(ibs_samples_alt)
        non_ibs_samples = y_df["ibd_i do not have this condition"][y_df["ibd_i do not have this condition"]==1].index.tolist()

        # sample_list = vegan_samples+non_vegan_samples[:len(vegan_samples)]
        sample_list = ibs_samples[:sample_num]+non_ibs_samples[:sample_num]
        y_df = y_df["ibd_diagnosed by a medical professional (doctor, physician assistant)"]
        y_df.loc[ibs_samples]=1
        y_df.loc[non_ibs_samples]=0
        y_df = y_df.loc[sample_list]
        self.y_df = y_df
        self.sample_list = sample_list
        self.X_df = self.asv_df.T #.loc[sample_list]
        ### match the indices of X and y
        self.match_Xy_df()


    def get_num_metadata_df(self):
        """ Translates metadata into a numeric dataframe using 1 hot encoding for each column"""
        num_metadata_df = pd.DataFrame()
        for col in self.metadata_df.columns[1:]:
            y_num_df = mb_utils.numeric_metadata_col(self.metadata_df, col)
            num_metadata_df = pd.concat([num_metadata_df, y_num_df],axis=1)
        
        drop_cols = [x for x in num_metadata_df.columns if "host_subject_id" in x or "survey_id_" in x or "anonymized_name" in x or "bmi_" in x or "collection_timestamp_" in x or "collection_time_" in x]
        num_metadata_df.drop(drop_cols,axis=1,inplace=True)
        self.num_metadata_df = num_metadata_df

        
        
    def run_micom_samples_parallel(self, com_model, processes=cpu_count(), atol=1e-6):
        """Function that parellizees MICOM with abundance data and other parameters
        ----------
        """
        compute_results_list = []
        # overlap_norm_df, all_norm_df = normalize_asv_abundances(genus_ASV, com_model)
        with ProcessPool(
            processes, 
            initializer=init_Com_Model, 
            initargs=(com_model, self.tradeoff_frac, atol, self.sample_medium_dict,
                      self.return_fluxes, self.pfba_bool, self.tradeoff_bool, self.min_growth_rates_bool, self.min_growth_df)
        ) as pool:
            try:
                ### FBA with specified objective coeeficients (cT)
                future = pool.map(
                    compute_comm_objective_task, 
                    [self.X_df.loc[x] for x in self.sample_list if self.X_df.loc[x].isna().values.any()==False], 
                    timeout=1800)
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
        if self.return_fluxes == True:
            self.compute_results = [x[:2] for x in compute_results_list]
            self.fluxes = [x[2] for x in compute_results_list]
            self.fluxes = pd.concat(self.fluxes,axis=1).T
            self.member_growth_rates = [x[3] for x in compute_results_list]
            self.member_growth_rates = pd.concat(self.member_growth_rates,axis=1).T
        else:
            self.compute_results = compute_results_list
        # return compute_results_list
        
    def set_sample_medium(self, randomize=True, max_flux=1000, num_zero=4, minimal_med_flux=1000):
        """
        First determines minimal media so models can at least grow. All samples will have these nutrients at 1000
        -------
        max_flux: The max intake fluxes possible (decides sampled flux range as well)
        num_zero: range for sampled integers. All non-zero entries will become 0, so larger num_zero means more zeros
        minimal_med_flux: flux for the nutrients in minimal media to undertake
        """
        med = minimal_medium(self.com_model, 1)
        if randomize==False:
            df = pd.DataFrame(max_flux, index=self.X_df.index, columns=self.com_model.medium.keys())
            for x in med.index:
                df[x] = minimal_med_flux
            self.sample_medium_dict = df.T.to_dict()
        else:
            df = pd.DataFrame(
                np.random.randint(0,max_flux,size=(len(self.X_df.index), len(self.com_model.medium.keys()))), 
                columns=self.com_model.medium.keys())
            df.index = self.X_df.index
            if num_zero!=0: # if 0, skip the setting of certain values to 0
                # Create Random Mask
                rand_zero_one_mask = np.random.randint(4, size=df.shape)
                # Fill df with 0 where mask is NOT 0
                df = df.where(rand_zero_one_mask==0, 0)
            for x in med.index:
                df[x] = minimal_med_flux
            self.sample_medium_dict = df.T.to_dict()


    def set_sample_medium_frac(self, randomize=True, max_flux=1, num_zero=4, flux_frac=0.001):
        """
        Constrains model to have very small inputs in comparison to original set_sample_medium function above
        -------
        max_flux: The max intake fluxes possible (decides sampled flux range as well)
        num_zero: range for sampled integers. All non-zero entries will become 0, so larger num_zero means more zeros
        minimal_med_flux: flux for the nutrients in minimal media to undertake
        """
        med = minimal_medium(self.com_model, 1)
        if randomize==False:
            df = pd.DataFrame(max_flux, index=self.X_df.index, columns=self.com_model.medium.keys())
            for x in med.index:
                df[x] = dict(med)[x]
            self.sample_medium_dict = df.T.to_dict()
        else:
            df = pd.DataFrame(
                np.random.randint(0,max_flux,size=(len(self.X_df.index), len(self.com_model.medium.keys()))), 
                columns=self.com_model.medium.keys())
            df = df*flux_frac
            df.index = self.X_df.index
            if num_zero!=0: # if 0, skip the setting of certain values to 0
                # Create Random Mask
                rand_zero_one_mask = np.random.randint(4, size=df.shape)
                # Fill df with 0 where mask is NOT 0
                df = df.where(rand_zero_one_mask==0, 0)
            for x in med.index:
                df[x] = dict(med)[x]
            self.sample_medium_dict = df.T.to_dict()

    def plot_pca_fluxes(self, n_components=3, plot_comps=[0,1], target_names=["non-vegan", "vegan"]):
        """Fits and plots PCA of the two dimensions
        Plot_comps specifies which components to drop
        target_names correspond to [0, 1], default just the vegan
        """
        X = self.fluxes
        y = self.y_df

        n_components = 3

        pca = PCA(n_components=n_components)
        X_pca = pca.fit_transform(X)

        colors = ['darkorange', 'turquoise'] # "navy"
        print(pca.explained_variance_ratio_)
        for X_transformed, title in [(X_pca, "PCA")]:
            plt.figure(figsize=(4, 4))
            for color, i, target_name in zip(colors, [0,1], target_names):
                plt.scatter(X_transformed[y == i, plot_comps[0]], X_transformed[y == i, plot_comps[1]],
                            color=color, lw=2, label=target_name)

            plt.legend(loc="best", shadow=False, scatterpoints=1)
            # plt.axis([-1, 1, -1, 1])
        plt.show()
        return pca, X_pca


    def get_min_growth_df(self):
        """ Solves for the min growth using cooperative trade-off with specified fraction"""
        # min_growth_dict = {}
        # Need to set X_df beforehand, and self.sample_list before...
        tradeoff = self.tradeoff_frac
        print("...setting min growth rates with tradeoff=%f"%(tradeoff))
        min_growth_df = pd.DataFrame()
        for abundance_series_df in [self.X_df.loc[x] for x in self.sample_list if self.X_df.loc[x].isna().values.any()==False]:
            print(abundance_series_df.name)
            with self.com_model as com_model_context:
                com_model_context = init_sample_com(abundance_series_df, com_model_context)
                sol = com_model_context.cooperative_tradeoff(fraction=tradeoff)
                sample_min_growth_series = sol.members.growth_rate.drop("medium")
                sample_min_growth_series.name = abundance_series_df.name
                # min_growth_dict.update({abundance_series_df.name, sol.members.growth_rate.drop("medium")})
                min_growth_df = pd.concat([min_growth_df, pd.DataFrame(sample_min_growth_series)],axis=1)
        self.min_growth_df = min_growth_df

    def set_react_objective(self, react, direction="max"):
        obj_rxn = self.com_model.reactions.get_by_id(react).forward_variable - self.com_model.reactions.get_by_id(react).reverse_variable
        self.com_model.objective = obj_rxn
        self.com_model.objective.direction = direction
        # gut_data = set_react_objective(gut_data, 'EX_ocdca_m', direction="max")

    def set_community_growth_objective(self):
        self.com_model.objective =  self.com_model.variables.community_objective

    
    def load_data(self,
                  FILE_COMM_MODEL = "../data/reconstructions/community_5_TOP-vegan.pickle",
                  FILE_GENUS_ASVS = "../data/agp_data/taxon_genus_asvs.csv",
                  FILE_METADATA = "../tables/mcdonald_agp_metadata.txt",
                  DIR_SIM_DATA = "../data/micom-sim-data/",
                  FILE_FLUX_DF = "micom_medium-fluxes-top50-9285_samples_fd.csv",
                  verbose=True
    ):
        self.dir_sim_data = DIR_SIM_DATA
        self.com_model = load_pickle(FILE_COMM_MODEL)
        self.fix_com_model(verbose)
        
        self.asv_df = pd.read_csv(FILE_GENUS_ASVS,index_col=0)

        com_objective = self.com_model.variables.community_objective
        
        if not os.path.exists(DIR_SIM_DATA):
            os.makedirs(DIR_SIM_DATA)
            print("The new directory (%s) is created!"%(DIR_SIM_DATA))

        if ".txt" in FILE_METADATA:
            # self.metadata_df = pd.read_csv("../tables/mcdonald_agp_metadata.txt",sep="\t")
            self.metadata_df = pd.read_csv(FILE_METADATA,sep="\t")
        else:
            # self.metadata_df = pd.read_csv("../metadata/metadata_biosample_filtered.csv",dtype=str)
            self.metadata_df = pd.read_csv(FILE_METADATA,dtype=str)

        self.metadata_df = self.metadata_df.set_index("sample_name",drop=True)

        # Get intersection of both asv_df and metadata_df and create new dfs with those indices
        samples_intersect = list(set(self.asv_df.columns).intersection(set(self.metadata_df.index)))
        # self.metadata_df = self.metadata_df.loc[self.asv_df.columns] #.copy()
        self.asv_df = self.asv_df[samples_intersect]
        self.asv_df_original = self.asv_df

        self.metadata_df = self.metadata_df.loc[samples_intersect]
        
        self.X_flux = pd.read_csv(self.dir_sim_data+FILE_FLUX_DF,index_col=0, low_memory=False)
        self.X_flux.index = self.X_flux.index.astype(str)
        self.X_flux = mb_utils.drop_constant_cols(self.X_flux)
        
        X_flux_consumed = self.X_flux[self.X_flux.columns[self.X_flux.mean()<0]].copy()
        self.X_flux_consumed_cols = [x.replace("EX_", "").replace("_m__medium", "[e]") for x in X_flux_consumed.columns]
        # print("len(X_flux_consumed_cols):",len(X_flux_consumed_cols))
        
        
        
Com_Model_Global=None
Tradeoff_Global=None
Atol_Global=None
Obj_Dict_Global=None
Return_Fluxes_Global=None
def init_Com_Model(mod, tradeoff, atol, sample_medium, return_fluxes, pfba_bool, tradeoff_bool, min_growth_rates_bool, min_growth_rates_df):
    global Com_Model_Global
    Com_Model_Global = mod
    del mod
    
    global Tradeoff_Global
    Tradeoff_Global = tradeoff
    del tradeoff
    
    global Atol_Global
    Atol_Global = atol
    del atol
    
    global Return_Fluxes_Global
    Return_Fluxes_Global = return_fluxes
    del return_fluxes
    
    global pFBA_Global
    pFBA_Global = pfba_bool
    del pfba_bool

    global Perform_Coop_Tradeoff
    Perform_Coop_Tradeoff = tradeoff_bool
    del tradeoff_bool
    
    global Min_Growth_Global
    Min_Growth_Global = min_growth_rates_bool
    del min_growth_rates_bool
    
    global Min_Growth_Df_Global
    Min_Growth_Df_Global = min_growth_rates_df
    del min_growth_rates_df
    
    global Sample_Medium_Global
    Sample_Medium_Global = sample_medium
    del sample_medium
    
    
def init_sample_com(abundance_series, com):
    """ Initializese the community model with the abundance data
    """
    com.set_abundance(abundance_series,normalize=True)
    com.id = abundance_series.name
    return com


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
    
    # 2. Apply min growth rates if True
    if Min_Growth_Global==True:
        _apply_min_growth(com_model_context, Min_Growth_Df_Global[abundance_series_df.name])
        
    # 3. Apply sample-specific media condition
    com_model_context.medium = Sample_Medium_Global[abundance_series_df.name]
    
    # 4. Perform FBA
    if Perform_Coop_Tradeoff == True:
        sol = com_model_context.cooperative_tradeoff(
            fluxes=Return_Fluxes_Global,
            pfba=pFBA_Global,
            fraction=Tradeoff_Global,
            atol=Atol_Global,
            rtol=None
        )
    else:
        sol = com_model_context.optimize(
            fluxes=Return_Fluxes_Global,
            pfba=pFBA_Global,
            raise_error=False,
            atol=Atol_Global,
            rtol=None
        )
    
    if Return_Fluxes_Global==True:
        # print(sol.objective_value)
        # res = get_micom_results(com_model_context, sol, tradeoff=Tradeoff_Global, atol=Atol_Global)
        res = get_micom_results_fluxes(com_model_context, sol, tradeoff=Tradeoff_Global, atol=Atol_Global)
        
        # return res
        flux_df = res.exchanges
        flux_df["react"] = flux_df["reaction"]+"__"+flux_df["taxon"]
        flux_df.set_index("react",inplace=True)
        flux_df = flux_df["flux"]
        flux_df.name = abundance_series_df.name

        # also return member growth rates
        grates = res.growth_rates["growth_rate"]
        grates.name = abundance_series_df.name
        # return (abundance_series_df.name, sol.growth_rate, flux_df)
        if Perform_Coop_Tradeoff == True:
            return (abundance_series_df.name, sol.growth_rate, flux_df, grates)
        else:
            return (abundance_series_df.name, sol.objective_value, flux_df, grates)
    else:
        # print(sol.objective_value)
        # if sol.growth_rate
        # return (abundance_series_df.name, sol.growth_rate)
        if Perform_Coop_Tradeoff == True:
            return (abundance_series_df.name, sol.growth_rate)
        else:
            return (abundance_series_df.name, sol.objective_value)
        

def get_micom_results_fluxes(com, sol, tradeoff=1, atol=1e-6):
    """ Returns fluxes of ALL reactions, not just exchanges"""
    ex_ids = [r.id for r in com.exchanges]
    rates = sol.members
    rates["taxon"] = rates.index
    rates["tradeoff"] = tradeoff
    rates["sample_id"] = com.id
    
    # exs = list({r.global_id for r in com.internal_exchanges + com.exchanges})
    # fluxes = sol.fluxes.loc[:, exs].copy()
    fluxes = sol.fluxes.copy()
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
    # exchanges["metabolite"] = anns.loc[exchanges.reaction, "metabolite"].values
    # exchanges["direction"] = DIRECTION[(exchanges.flux > 0.0).astype(int)].values
    # exchanges = exchanges[exchanges.flux.abs() > exchanges.tolerance]

    return GrowthResults(growth, exchanges, anns)