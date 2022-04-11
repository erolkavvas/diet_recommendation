import sys
sys.path.append('../')
import pandas as pd
import nevergrad as ng
import time
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
# sns.set_theme(style="ticks")

### mb_xai packages
from mb_xai import mb_utils
import mb_xai.mb_data as gd
# from mb_xai.micom_learn import run_micom_samples_parallel, init_Com_Model, compute_comm_objective_task

from multiprocessing import cpu_count
from pebble import ProcessPool, ThreadPool
from concurrent.futures import TimeoutError

from sklearn.metrics import roc_curve, auc, log_loss, accuracy_score
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import plot_precision_recall_curve, precision_recall_curve
from sklearn.metrics import PrecisionRecallDisplay, RocCurveDisplay
from sklearn.metrics import average_precision_score

# importing micom opens INFO logging
from micom.annotation import annotate_metabolites_from_exchanges
from micom.workflows.core import workflow, GrowthResults
from micom.util import _apply_min_growth
import logging
logging.getLogger("micom").setLevel(logging.WARNING)

class GutLearn(object):
    """ Class object holding information about Gut project dataset
        - Performs ML and statistical tests
    """
    def __init__(self):
        self.com_model = None
        self.X_df = pd.DataFrame()
        self.y_df = pd.DataFrame()
        self.sample_list = []
        self.coef_params = []
        self.coef_params_list = []
        self.obj_var_list = []
        self.obj_react_list = []
        self.iter_tracker_text = None
        self.write_iter_tracker = True
        self.return_fluxes = False
        self.run_obj_params = True
        self.pfba_bool = True
        self.tradeoff_bool = False
        self.tradeoff_frac = 0.3 # Micom recommends tradeoff frac<0.5 for relevant growth rates, used when min_growth_rates_bool=True or when tradeoff_bool=True
        self.coef_norm_type="l2" # could also be "l1"
        self.alpha = 0.3 # weight of regularization penalty
        self.compute_results = None
        self.ng_coef_params = None
        self.coef_params = None
        self.y_sigmiod_probs = None
        self.y_sigmiod_probs_list = []
        self.compute_results_list = []
        self.loss_list = []
        self.accuracy_list = []
        self.flip_probs = True
        self.pr_roc_dict = {}
        self.recommendation = None
        self.sample_weight = None
        self.fluxes = None
        self.member_growth_rates = None
        self.iteration_i = 0
        self.equal_forward_reverse_coef = True
        self.min_growth_rates_bool = False # constraint the models according to their minimum growth rate determined by cooperative tradeoff
        self.min_growth_df = pd.DataFrame()
        
        
    def ng_learn_micom_params(self, coef_params):
        """Function to be minimized by nevergrad, differnt coefficients are input
        that result in different values"""
        # create objective dict with params
        self.iteration_i = self.iteration_i + 1

        # set the forward and reverse variable to have the same coef
        if self.equal_forward_reverse_coef==True:
            obj_react_dict = dict(zip(self.obj_react_list, coef_params))
            init_obj_var_dict = {}
            for rxn, coef in obj_react_dict.items():
                init_obj_var_dict.update({rxn.forward_variable: coef})
                init_obj_var_dict.update({rxn.reverse_variable: -coef})
        else:
            init_obj_var_dict = dict(zip(self.obj_var_list, coef_params)) # old way

        start = time.process_time() # time stamp not working (~5mins for 50genera model)
        ### Not sure why but sometimes some models will result in None??
        compute_results_ret = self.run_micom_samples_parallel(
            self.com_model, init_obj_var_dict,
            processes=cpu_count(), 
            tradeoff=self.tradeoff_frac, 
            atol=None # 1e-6
        )
        ### Tuple is different depending on if fluxes are returned or not
        if self.return_fluxes ==True:
            self.compute_results = [x[:2] for x in compute_results_ret]
            self.fluxes = [x[2] for x in compute_results_ret]
            self.fluxes = pd.concat(self.fluxes,axis=1).T
            self.member_growth_rates = [x[3] for x in compute_results_ret]
            self.member_growth_rates = pd.concat(self.member_growth_rates,axis=1).T
        else:
            self.compute_results = compute_results_ret
        self.coef_params = coef_params
        self.coef_params_list.append(coef_params)
        
        if compute_results_ret!=None: # None is the case where some samples dont compute
            # Compute loss using regularized Log Loss
            y_test_series = self.y_df
            y_score_series = pd.Series(dict(self.compute_results))
            y_score_array = y_score_series.values
            y_test_series = y_test_series.reindex(y_score_series.index)
            y_test_array = y_test_series.values
            
            # Standardize computed values and convert to sigmoid probs
            y_sigmoid_probs = get_sigmoid_prob(y_test_series, y_score_series)
            
            ## Log loss, ROC, Accuracy
            loss_log = log_loss(y_test_array, y_sigmoid_probs, normalize=True, sample_weight=self.sample_weight)
            loss_log_reflect = log_loss(y_test_array, 1-y_sigmoid_probs, normalize=True, sample_weight=self.sample_weight)
            if self.flip_probs == True:
                if loss_log_reflect<loss_log: # sometimes great seperation happens, but the probabilities are opposite...
                    y_sigmoid_probs = 1-y_sigmoid_probs
                    loss_log = loss_log_reflect
            
            ## Get PR and ROC insights (any difference between y_test_array and self.y_df?)
            fpr_, tpr_, _ = roc_curve(y_test_array, y_sigmoid_probs)
            roc_auc_ = auc(fpr_, tpr_)
            precision, recall, thresholds = precision_recall_curve(y_test_array, y_sigmoid_probs)
            AP = average_precision_score(y_test_array, y_sigmoid_probs)
            self.pr_roc_dict.update({
                "fpr": fpr_, "tpr": tpr_, "auc": roc_auc_, "precision": precision, "recall": recall, "AP": AP})
            
            self.y_sigmiod_probs = y_sigmoid_probs         
            y_pred = self.y_sigmiod_probs.copy() # otherwise y_sigmoid_probs becomes y_pred
            y_pred[y_pred <= 0.5] = 0
            y_pred[y_pred > 0.5] = 1

            accuracy = accuracy_score(y_test_array, y_pred)
            self.accuracy_list.append(accuracy)
            self.y_sigmiod_probs_list.append(y_sigmoid_probs)
            self.compute_results_list.append(compute_results_ret)
            
            if self.coef_norm_type=="l1":
                vec_ = np.abs(coef_params)
                regularization_cost = np.sum(vec_)
            elif self.coef_norm_type=="l2":
                regularization_cost = 0.5*np.dot(coef_params.T, coef_params)

            minimization_error = loss_log + self.alpha*regularization_cost

            iteration_time = time.process_time() - start
            print("%d) Acc: %.2f, AUC: %.2f, LogLoss: %.5f, %s-Norm: %.2f, Error: %.5f, time: %f"%(
                self.iteration_i, accuracy, roc_auc_, loss_log, self.coef_norm_type, regularization_cost, minimization_error, iteration_time))
            if self.write_iter_tracker == True:
                write_out = "%d, %.5f, %.5f, %.5f, %.5f, %.5f, %.5f"%(
                    self.iteration_i, accuracy, roc_auc_, loss_log, regularization_cost, minimization_error, iteration_time)
                self.iter_tracker_text.write(write_out+"\n")

        else:
            minimization_error = 100
            print("Error(failed micom run): %f,"%(minimization_error))
            
        self.loss_list.append(minimization_error)
            
        return minimization_error
    
    
    def run_micom_samples_parallel(self, 
        com_model, objective_dict, 
        processes=cpu_count(),
        tradeoff=0.3, 
        atol=1e-6

    ):
        """Function that performs MICOM with abundance data and other parameters
        ----------
        objective_dict : dict
            dictionary with forward and reverse reaction variables as key and their objective coefficients as items
        """
        compute_results_list = []
        # overlap_norm_df, all_norm_df = normalize_asv_abundances(genus_ASV, com_model)
        with ProcessPool(
            processes, 
            initializer=init_Com_Model, 
            initargs=(com_model, tradeoff, atol, objective_dict, self.return_fluxes, self.pfba_bool, self.tradeoff_bool, self.min_growth_rates_bool, self.min_growth_df)
        ) as pool:
            try:
                ### FBA with specified objective coeeficients (cT)
                future = pool.map(
                    compute_comm_objective_task, 
                    [self.X_df.loc[x] for x in self.sample_list if self.X_df.loc[x].isna().values.any()==False], 
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
        

    def set_sample_weight(self):
        """Balances sample cost to frequency of class
           Same as "balanced" in sklearn 
           (https://www.analyticsvidhya.com/blog/2020/10/improve-class-imbalance-class-weights/)
        """
        n_classes = len(self.y_df.unique())
        n_samples_0 = self.y_df.value_counts()[0]
        n_samples_1 = self.y_df.value_counts()[1]
        n_samples = self.y_df.count()
        w_0 = n_samples/(n_classes*n_samples_0)
        w_1 = n_samples/(n_classes*n_samples_1)
        class_weights = {0: w_0, 1: w_1}
        sample_weight = self.y_df.copy()
        sample_weight = sample_weight.map(class_weights)
        self.sample_weight = sample_weight
    

    def plot_opt_probs_trajectory(self, figsize=(10,5)):
        """Plots trajectories of optimum values and probablities to see how seperation progresses"""
        traj_opt_df = get_opt_traj(self.compute_results_list)
        traj_opt_melt_df = melt_list_df(traj_opt_df, self)
        f, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=figsize)
        g1 = sns.lineplot(x="iterations", y="value",hue="trait",style="index",data=traj_opt_melt_df,legend=False,ax=ax1)
        g1.axes.set_title("Optimum values trajectory")

        traj_prob_df = pd.DataFrame(self.y_sigmiod_probs_list, columns=self.sample_list).T
        traj_prob_melt_df = melt_list_df(traj_prob_df, self)
        # traj_prob_df[:10].T.plot()
        g2 = sns.lineplot(x="iterations", y="value",hue="trait",style="index",data=traj_prob_melt_df,legend=False,ax=ax2)
        g2.axes.set_title("Sigmoid probabilities trajectory")
        # f, (ax1, ax2) = plot_opt_probs_trajectory(self, figsize=(10,5))
        return f, (ax1, ax2)
    
    
    def output_figures(self, save=True):
        """Output figures from a simulation into self.dir_sim_data folder"""
        ### Trajectory of optimum values and probabilities
        f, (ax1, ax2) = self.plot_opt_probs_trajectory(figsize=(13,4))
        if save==True:
            f.savefig(self.dir_sim_data+"opt_probs_trajectory.png")
            f.savefig(self.dir_sim_data+"opt_probs_trajectory.svg")

        ### Trajectory of accuracy and loss
        f, (ax1, ax2) = plt.subplots(1, 2, figsize=(12,4))
        ax1.plot(range(len(self.accuracy_list)), self.accuracy_list)
        ax1.set_xlabel("Iterations")
        ax1.set_ylabel("Accuracy")
        ax2.plot(range(len(self.loss_list)), self.loss_list)
        ax2.set_xlabel("Iterations")
        ax2.set_ylabel("loss")
        if save==True:
            f.savefig(self.dir_sim_data+"accuracy_loss_trajectory.png")
            f.savefig(self.dir_sim_data+"accuracy_loss_trajectory.svg")

        if type(self.recommendation)==np.ndarray:
            self.write_iter_tracker = False
            self.ng_learn_micom_params(self.recommendation)

            f, (ax1, ax2) = plt.subplots(nrows=1, ncols=2,figsize=(10,4))
            roc_display = RocCurveDisplay(fpr=self.pr_roc_dict["fpr"], tpr=self.pr_roc_dict["tpr"])
            roc_display.plot(ax=ax1, label='Metabolic Network AUC={0:0.2f}'.format(self.pr_roc_dict["auc"]))
            pr_display = PrecisionRecallDisplay(precision=self.pr_roc_dict["precision"], recall=self.pr_roc_dict["recall"])
            pr_display.plot(ax=ax2, label='Metabolic Network AP={0:0.2f}'.format(self.pr_roc_dict["AP"]))
            if save==True:
                f.savefig(self.dir_sim_data+"roc_pr_curves.png")
                f.savefig(self.dir_sim_data+"roc_pr_curves.svg")

            f, ax = plt.subplots(nrows=1, ncols=1,figsize=(5,4))
            ax.hist(self.recommendation)
            ax.set_ylabel("Count")
            ax.set_xlabel("Coefficient value")
            f.savefig(self.dir_sim_data+"coefficient_distributions.png")
            f.savefig(self.dir_sim_data+"coefficient_distributions.svg")
    

    def save_sim_params(self):
        save_list = [
            "sample_list", "alpha", "coef_norm_type",'coef_params', "pfba_bool", "coef_norm_type",
            'loss_list',"accuracy_list",'flip_probs','recommendation', "min_growth_rates_bool", "tradeoff_frac"]
        save_tuple = []
        for x in save_list:
            save_tuple.append((x, self.__dict__[x]))
        save_tuple = tuple(save_tuple)
        with open(self.dir_sim_data+'sim_params.pickle', 'wb') as f: ## save recommendation to file
            pickle.dump(save_tuple, f)


    def load_sim_params(self):
        """Loads a simulated run and initializes the object accordingly
        """
        with open(self.dir_sim_data+'sim_params.pickle', 'rb') as f:
            sim_data = pickle.load(f)
            
        sim_data = dict(sim_data)
        self.alpha = sim_data['alpha'] # no L1 or L2 penalty if alpha=0
        self.coef_norm_type = sim_data["coef_norm_type"]
        self.flip_probs = sim_data["flip_probs"] 
        self.recommendation = sim_data["recommendation"]
        self.tradeoff_frac = sim_data["tradeoff_frac"]
        self.sample_list = sim_data["sample_list"] 
        self.min_growth_rates_bool = sim_data["min_growth_rates_bool"]
        ## I should also load in the X_train, Y_train, X_test, Y_test??


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
    

    def init_exch_bounds(self, ub=50, lb=-50): # note necessary code
        for x in self.com_model.exchanges:
            x.lower_bound = lb
            x.upper_bound = ub
            
            
    def get_min_growth_df(self, tradeoff):
        """ Solves for the min growth using cooperative trade-off with specified fraction"""
        # min_growth_dict = {}
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
    
    
    def set_GutData(self, GutData_obj, class_weight=None, obj_vars="forward", lower=-10, upper=10, DIR_SIM_DATA="../sim-data/"):
        """Initialize using GutData object
            class_weight: None, "balanced". balanced alters sample loss the same as sklearn "balanced"
            obj_vars: "foward", "reverse", "both" - decideds the rxn variables to be included
            lower: lower limit on objective coefficients
            upper: upper limit on objective coefficients
        
        """
        self.com_model = GutData_obj.com_model
        self.X_df = GutData_obj.X_df
        self.y_df = GutData_obj.y_df
        self.sample_list = GutData_obj.sample_list

        # com_objective = GutData_obj.com_model.objective.variables
        com_objective = self.com_model.variables.community_objective
        
        if class_weight=="balanced": # setting class weights
            self.set_sample_weight()
            
        if obj_vars=="forward":
            self.obj_var_list = [x.forward_variable for x in self.com_model.exchanges]
            self.ng_coef_params = ng.p.Array(shape=((len(self.obj_var_list)),))
            self.ng_coef_params.set_bounds(lower=0, upper=upper)
        elif obj_vars=="reverse":
            self.obj_var_list = [x.reverse_variable for x in self.com_model.exchanges]
            self.ng_coef_params = ng.p.Array(shape=((len(self.obj_var_list)),))
            self.ng_coef_params.set_bounds(lower=lower, upper=0)
        elif obj_vars=="both":
            self.obj_var_list = [x.reverse_variable for x in self.com_model.exchanges]
            for x in self.com_model.exchanges:
                self.obj_var_list.append(x.forward_variable)

            self.obj_react_list = [x for x in self.com_model.exchanges] # new line
            if self.equal_forward_reverse_coef==True:
                self.ng_coef_params = ng.p.Array(shape=((len(self.obj_react_list)),))
            else:
                self.ng_coef_params = ng.p.Array(shape=((len(self.obj_var_list)),))
                
            self.ng_coef_params.set_bounds(lower=lower, upper=upper)

        elif obj_vars=="growth": # set objective to just be community objective
            # self.obj_var_list = list(com_objective)
            self.obj_var_list = [com_objective]
            self.ng_coef_params = ng.p.Array(shape=((len(self.obj_var_list)),))
            self.ng_coef_params.set_bounds(lower=0, upper=upper)
        elif type(obj_vars)!=str: # manually set objective to be obj_vars (list of reaction variables)
            self.obj_react_list = obj_vars
            for i in self.obj_react_list:
                self.obj_var_list.append(i.forward_variable)
                self.obj_var_list.append(i.reverse_variable)
            self.ng_coef_params = ng.p.Array(shape=((len(self.obj_react_list)),))
            self.ng_coef_params.set_bounds(lower=lower, upper=upper)
        
        obj_rxn = 0
        for i in self.obj_var_list:
            obj_rxn+=i
        self.com_model.objective = obj_rxn
        
        if self.min_growth_rates_bool == True:
            self.get_min_growth_df(self.tradeoff_frac)
        
        if not os.path.exists(DIR_SIM_DATA):
            os.makedirs(DIR_SIM_DATA)
            print("The new directory (%s) is created!"%(DIR_SIM_DATA))
        
        self.dir_sim_data = DIR_SIM_DATA
        if self.write_iter_tracker==True:
            self.iter_tracker_text = open(DIR_SIM_DATA+"iter_tracker.txt","w")
            self.iter_tracker_text.write("loss_roc_auc, loss_log, regularization_cost, minimization_error, iteration_time\n")
        
        


Com_Model_Global=None
Tradeoff_Global=None
Atol_Global=None
Obj_Dict_Global=None
Return_Fluxes_Global=None
def init_Com_Model(mod, tradeoff, atol, obj_dict, return_fluxes, pfba_bool, tradeoff_bool, min_growth_rates_bool, min_growth_rates_df):
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

    global Perform_Coop_Tradeoff
    Perform_Coop_Tradeoff = tradeoff_bool
    del tradeoff_bool
    
    global Min_Growth_Global
    Min_Growth_Global = min_growth_rates_bool
    del min_growth_rates_bool
    
    global Min_Growth_Df_Global
    Min_Growth_Df_Global = min_growth_rates_df
    del min_growth_rates_df

    
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
    
    # 2. Set objective function coefficients
    com_model_context.objective.set_linear_coefficients(Obj_Dict_Global)
    
    if Min_Growth_Global==True:
        _apply_min_growth(com_model_context, Min_Growth_Df_Global[abundance_series_df.name])
    
    # 3. Perform FBA
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
        # if sol.growth_rate
        # return (abundance_series_df.name, sol.growth_rate)
        if Perform_Coop_Tradeoff == True:
            return (abundance_series_df.name, sol.growth_rate)
        else:
            return (abundance_series_df.name, sol.objective_value)
    
    
def get_scaled_vec(y_score_array):
    """Returns both L2 norm scaling of data and regular Standardscaling"""
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


def get_sigmoid_prob(y_test_series, y_score_series):
    """ Tranforms opt values to sigmoid probabilities
    (1) standard scales the outputs (0 mean, 1 std)
    (2) sigmoid activation 1/(1+e^-y)
    """
    y_score_array = y_score_series.values
    y_test_series = y_test_series.reindex(y_score_series.index)
    y_test_array = y_test_series.values
    y_score_Norm, y_score_StdScale = get_scaled_vec(y_score_array)
    y_score_sigmoid_probs = 1/(1 + np.exp(-y_score_StdScale))
    return y_score_sigmoid_probs
    
    
def get_micom_results(com, sol, tradeoff=1, atol=1e-6):
    """ Returns just exchanges"""
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


def get_opt_traj(gut_learn_list):
    """Returns a df with columns iterations and rows samples. values are opt values"""
    results_list_df = pd.DataFrame()
    for i, df in enumerate(gut_learn_list):
        df_iter = pd.DataFrame.from_dict(dict(df),orient="index", columns=[i])
        results_list_df = pd.concat([results_list_df, df_iter],axis=1)
    
    return results_list_df

# traj_opt_df = get_opt_traj(gut_learn.compute_results_list)
# traj_opt_df[:10].T.plot()


def melt_list_df(list_df, gut_learn):
    df = list_df.reset_index()
    df["trait"] = df["index"].map(gut_learn.y_df)
    df = df.melt(id_vars=["index", "trait"],var_name="iterations")
    return df