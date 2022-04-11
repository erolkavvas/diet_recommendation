# coding: utf-8
### -------------------------------------------------------------------------
### mb_data.py  
### Erol Kavvas, UCD, 2021 
### -------------------------------------------------------------------------
###     "ensemble.py" provides a class object for computing with a population 
### --------------------------------------------------------------------------
import numpy as np
import pandas as pd
from os import listdir, path
from collections import Counter
from tqdm import tqdm
import sys
sys.path.append('../')

### mb_xai packages
from mb_xai import mb_utils
# from mb_xai import micom_learn

### Skbio
from skbio.stats.composition import ancom
from skbio.stats.composition import multiplicative_replacement

## Plotting
import matplotlib.pyplot as plt

## Sklearn
import sklearn
from sklearn.feature_selection import f_classif
from sklearn.model_selection import StratifiedShuffleSplit, RepeatedStratifiedKFold
from sklearn.model_selection import permutation_test_score, cross_val_score
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, auc, accuracy_score, matthews_corrcoef
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.decomposition import PCA
from sklearn.metrics import plot_precision_recall_curve, precision_recall_curve
from sklearn.metrics import PrecisionRecallDisplay, RocCurveDisplay
from sklearn.metrics import average_precision_score
from sklearn.model_selection import StratifiedKFold
# imblearn SMOTE
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.pipeline import make_pipeline

### Micom
from micom import load_pickle

class GutData(object):
    """ Class object holding information about Gut project dataset
        - Performs ML and statistical tests
    """
    def __init__(self):
        self.asv_df = pd.DataFrame()
        self.asv_df_original = pd.DataFrame()
        self.metadata_df = pd.DataFrame()
        self.com_model = None
        self.X_df = pd.DataFrame()
        self.y_df = pd.DataFrame()
        self.sample_list = []
        self.multiplicative_replacement=False
        self.rf = RandomForestClassifier(
            n_estimators=512, min_samples_leaf=1, n_jobs=-1, bootstrap=True, max_samples=0.7, class_weight='balanced')
        self.logreg = LogisticRegression(
            C=1e2,fit_intercept=True,intercept_scaling=True,solver='liblinear', penalty="l2", class_weight='balanced',max_iter=1000) 
        self.nn = MLPClassifier(
            alpha=1e-4, activation="logistic", hidden_layer_sizes=(5, 2), solver='lbfgs',max_iter=1000)
        
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
        
        
    def fix_com_model(self, verbose=True):
        """ Manually fixes minor issues in provided micom community metabolic models"""
        try:
            self.com_model.reactions.get_by_id("tDHNACOA__lactobacillus").id = "EX_tDHNACOA(e)__lactobacillus"
            self.com_model.reactions.get_by_id("EX_tDHNACOA(e)__lactobacillus").build_reaction_from_string('dhnacoa[e]__lactobacillus <=> 0.0023125206475057814 dhnacoa_m')
            if verbose==True:
                print("Fixed EX_tDHNACOA(e)")
        except:
            if verbose==True:
                print("Nothing to fix")
        # return com_model
        
        
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
        # return y_df, sample_list
        
        
    def run_smote_clf(self, clf=None, cv=None, use_smote=True):
        """Adds synthetic data for better classification performance
           if cv int, specifies number of stratified kfolds
        """
        if clf==None:
            clf=self.logreg
        
        if use_smote==True:
            model = make_pipeline(
                SMOTE(),
                # ADASYN(),
                clf)
        else:
            model = make_pipeline(
                # ADASYN(),
                clf)
        cv_results = cross_validate(
            model, self.X_df, self.y_df, cv=cv, scoring=["balanced_accuracy","accuracy"],
            return_train_score=True, return_estimator=True)
        for score_type in ["balanced_accuracy","accuracy"]:
            print(
                f"{score_type} mean +/- std. dev.: "
                f"{cv_results['test_'+score_type].mean():.3f} +/- "
                f"{cv_results['test_'+score_type].std():.3f}"
            )
        avg_coef = 0
        if use_smote==True:
            for est in cv_results["estimator"]:
                if type(clf)==sklearn.ensemble._forest.RandomForestClassifier:
                    avg_coef += est[1].feature_importances_
                elif type(clf)==sklearn.linear_model._logistic.LogisticRegression:
                    avg_coef += est[1].coef_
        else:
            for est in cv_results["estimator"]:
                if type(clf)==sklearn.ensemble._forest.RandomForestClassifier:
                    avg_coef += est[0].feature_importances_
                elif type(clf)==sklearn.linear_model._logistic.LogisticRegression:
                    avg_coef += est[0].coef_
        
        avg_coef = avg_coef/5
        if type(clf)==sklearn.ensemble._forest.RandomForestClassifier:
            lr_coef_df = pd.DataFrame(avg_coef, index=gut_data.X_df.columns, columns=["avg_coef"])
            lr_coef_df.sort_values("avg_coef", ascending=False,inplace=True)
        elif type(clf)==sklearn.linear_model._logistic.LogisticRegression:
            lr_coef_df = pd.DataFrame(avg_coef, index=["avg_coef"], columns=gut_data.X_df.columns).T
            lr_coef_df.sort_values("avg_coef", ascending=False,inplace=True)
        return clf, lr_coef_df
        
    
    def plot_pca(self, plot_comps=[0,1], target_names=["non-vegan", "vegan"]):
        """Fits and plots PCA of the two dimensions
        Plot_comps specifies which components to drop
        target_names correspond to [0, 1], default just the vegan
        """
        X = self.X_df
        y = self.y_df

        n_components = 3
        
        pca = PCA(n_components=n_components)
        X_pca = pca.fit_transform(X)

        colors = ['darkorange', 'turquoise'] # "navy"
        
        for X_transformed, title in [(X_pca, "PCA")]:
            plt.figure(figsize=(4, 4))
            for color, i, target_name in zip(colors, [0,1], target_names):
                plt.scatter(X_transformed[y == i, plot_comps[0]], X_transformed[y == i, plot_comps[1]],
                            color=color, lw=2, label=target_name)

            plt.legend(loc="best", shadow=False, scatterpoints=1)
            plt.axis([-1, 1, -1, 1])
        plt.show()
        
        
    def test_get_f_classif(self):
        """ performs sklearn ANOVA F-value for each feature and returns in order of ascending p-values"""
        f_classif_df = pd.DataFrame(f_classif(self.X_df, self.y_df),columns=self.X_df.columns, index=["f_stat", "p_val"])
        f_classif_df = f_classif_df.T
        # f_classif_df.columns = ["f_stat", "p_val"]
        f_classif_df.sort_values("p_val", ascending=True,inplace=True)
        return f_classif_df
    
    
    def test_get_ancom(self,delta=None,alpha=0.05,tau=0.02,theta=0.1,multiple_comparisons_correction='holm-bonferroni'):
        """ Performs ANCOM using scikit-bio"""
        if self.multiplicative_replacement==False:
            test_X_df = pd.DataFrame(multiplicative_replacement(self.X_df,delta=delta),index=gut_data.X_df.index,columns=self.X_df.columns)
        else:
            test_X_df = self.X_df
        ancom_df, percentile_df = ancom(
            test_X_df,self.y_df, alpha=alpha,tau=tau,theta=theta,multiple_comparisons_correction=multiple_comparisons_correction)
        ancom_df.sort_values("W", ascending=False,inplace=True)
        return ancom_df, percentile_df
    
    
    def train_model(self, X_train, X_test, y_train, y_test, model_type="lasso"):
        if model_type == "lasso":        
            # alg = LogisticRegression(solver='liblinear', penalty="l2", class_weight='balanced')    
            alg = self.logreg
            alg.fit(X_train, y_train)
            imp = alg.coef_[0,:]
        if model_type == "rf":
            alg = self.rf
            # alg = RandomForestClassifier()
#             alg = RandomForestClassifier(n_estimators=512, min_samples_leaf=1, n_jobs=-1, bootstrap=True, 
#                                          max_samples=0.7, class_weight='balanced')
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

    
    def run_trait_ml(self, n_splits=25, test_size=0.25, ml_type = "rf", scale=True, verbose=True):
        """ Runs Random forests or lasso on the specified trait similar to Nature study """
        # y_df = mb_utils.numeric_metadata_col(self.metadata_df, y_col, encode_type="one-hot")
        # y_df = mb_utils.drop_notprovided(y_df)
        X_df, y_df = mb_utils.get_x_y_train_test(self.X_df, self.y_df)
        # y_df = y_df[y_col_id]
        # if lognorm=True:
        if scale==True:
            X_df = np.log(X_df + 1.0)

        sss = StratifiedShuffleSplit(n_splits=n_splits, test_size=test_size)
        # sss = RepeatedStratifiedKFold(n_splits=3, n_repeats=3) #75/25 training/test split for each iteration
        importances, aucs, accs, matthews = [], [], [], []

        for train_index, test_index in tqdm(sss.split(X_df, y_df)):
            # print("TRAIN:", train_index, "TEST:", test_index)
            X_train, X_test = X_df.iloc[train_index], X_df.iloc[test_index]
            y_train, y_test = y_df.iloc[train_index], y_df.iloc[test_index]
            # print(Counter(y_train), Counter(y_test))
            if scale==True:
                scaler = StandardScaler().fit(X_train)
                X_train = pd.DataFrame(scaler.transform(X_train), index=X_train.index, columns=X_train.columns)
                X_test = pd.DataFrame(scaler.transform(X_test), index=X_test.index, columns=X_test.columns)
                
            # print(ml_type)
            imp, roc_auc, acc, matthew = self.train_model(X_train, X_test, y_train, y_test, model_type=ml_type)
            importances.append(imp)
            aucs.append(roc_auc)
            accs.append(acc)
            matthews.append(matthew)

        avg_importances = mb_utils.getImportances(importances, X_df.columns, self.y_df.name)
        if verbose==True:
            print("Mean AUC:%f, Mean Accuracy:%f, Mean Matthews:%f"%(np.mean(aucs), np.mean(accs), np.mean(matthews)))
        return avg_importances, aucs, accs, matthews
        
        
    def fit_baseline_logreg(self, verbose=True,cv=5, scale=False,penalty='l2',C=1.0,fit_intercept=True,intercept_scaling=1,class_weight=None,max_iter=1000):
        """Fits a baseline logistic regression model on the training data
           Experience shows that a high C (i.e., no regularization) gives best accuracy
        """
        if scale==True: # scale data to 0 mean and 0 variance (results dont really change)
            scaler = StandardScaler().fit(self.X_df)
            X_train = pd.DataFrame(scaler.transform(self.X_df), index=self.X_df.index, columns=self.X_df.columns)
        else:
            X_train = self.X_df
        clf = LogisticRegression(
            penalty=penalty,C=C,fit_intercept=fit_intercept,intercept_scaling=intercept_scaling,
            class_weight=class_weight, max_iter=max_iter
                            ).fit(X_train, self.y_df)
        lr_coef_df = pd.DataFrame(clf.coef_, index=["coef"], columns=gut_data.X_df.columns).T
        lr_coef_df.sort_values("coef", ascending=False,inplace=True)
        acc = clf.score(X_train, self.y_df)
        fpr_, tpr_, _ = roc_curve(self.y_df.values, clf.predict(X_train))
        roc_auc_ = auc(fpr_, tpr_)
        scores = cross_val_score(clf, X_train, self.y_df, cv=cv)
        if verbose==True:
            print("ROC-AUC:%f, Accuracy:%f, Mean CV accuracy:%f, Mean CV std:%f"%(roc_auc_, acc, scores.mean(), scores.std()))
        return clf, lr_coef_df
        
        
    def load_data(self,
                  FILE_COMM_MODEL = "../reconstructions/community_5_TOP-vegan.pickle",
                  FILE_GENUS_ASVS = "../tables/taxon_genus_asvs.csv",
                  DIR_SIM_DATA = "../sim-data/",
                  verbose=True
    ):
        self.com_model = load_pickle(FILE_COMM_MODEL)
        self.fix_com_model(verbose)
        
        self.asv_df = pd.read_csv(FILE_GENUS_ASVS,index_col=0)
        self.asv_df_original = self.asv_df
        
        if verbose==True:
            mb_utils.get_com_model_info(self.com_model)
        
        self.metadata_df = pd.read_csv("../tables/mcdonald_agp_metadata.txt",sep="\t")
        self.metadata_df = self.metadata_df.set_index("sample_name",drop=True)
        self.metadata_df = self.metadata_df.loc[self.asv_df.columns] #.copy()
        
        
#class MicomLearn(object):

### plotting functions for ROC and PR curves
def update_recall_precision_dict(recall, precision, recall_precision_dict):
    for i, val in enumerate(recall):
        if val in recall_precision_dict.keys():
            recall_precision_dict[val].append(precision[i])
        else:
            recall_precision_dict.update({val: [precision[i]]})
    return recall_precision_dict


def get_PR_ROC(model, skf):
    avg_AP, avg_AUC = 0, 0
    recall_precision_dict, tpr_fpr_dict = {}, {}
    avg_recall, avg_precision, avg_thresh,  = [],[],[]
    for train_index, test_index in skf.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        model.fit(X_train, y_train)
        
        precision, recall, thresholds = precision_recall_curve(y_test, model[1].predict_proba(X_test)[:,1])
        recall_precision_dict = update_recall_precision_dict(recall, precision, recall_precision_dict)
        AP = average_precision_score(y_test, model[1].predict_proba(X_test)[:,1])

        fpr, tpr, _ = roc_curve(y_test, model[1].predict_proba(X_test)[:,1])
        roc_auc = auc(fpr, tpr)
        tpr_fpr_dict = update_recall_precision_dict(tpr, fpr, tpr_fpr_dict)
    
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
    
    return recall_vals, precision_vals, avg_AP, tpr_vals, fpr_vals, avg_AUC


def get_baseline(y_df, val_binary=0):
    """Say all values are 0"""
    baseline_pred_df = pd.Series(data=val_binary,index=y_df)
    
    acc = accuracy_score(y_df, baseline_pred_df)
    
    precision, recall, thresholds = precision_recall_curve(y_df, baseline_pred_df)
    AP = average_precision_score(y_df, baseline_pred_df)

    fpr, tpr, _ = roc_curve(y_df, baseline_pred_df)
    roc_auc = auc(fpr, tpr)
    
    return recall, precision, AP, tpr, fpr, roc_auc