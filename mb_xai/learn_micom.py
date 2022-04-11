import sklearn
import numpy as np
import heapq
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler, QuantileTransformer, quantile_transform
from sklearn.model_selection import cross_validate, train_test_split
from sklearn.multioutput import MultiOutputRegressor
from sklearn.linear_model import RidgeCV
from sklearn.compose import TransformedTargetRegressor
from sklearn.metrics import median_absolute_error, r2_score
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression

class LearnMicom(object):
    """ ML functionality for learning MICOM fluxes
    
    Note - scaling all the data before fitting model is bad practice and causes data leakage
    """
    def __init__(self):
        self.X_df = pd.DataFrame()
        self.y_df = pd.DataFrame()
        self.orig_X_df = pd.DataFrame()
        self.orig_y_df = pd.DataFrame()
        # self.nn = MLPRegressor(hidden_layer_sizes=(32,32,32,32), max_iter=10000, alpha=1e4)
        self.nn = MLPRegressor(hidden_layer_sizes=(100,100,100,100), max_iter=10000, alpha=1)
        self.input_scaler = None
        self.output_scaler = None
        self.medium_cols = []
        
    def scale_inputs(self):
        """Highly recommended to standardize inputs for a Neural Net"""
        scaler = StandardScaler()
        scaler.fit(self.X_df)
        self.X_df = scaler.transform(self.X_df)
        self.X_df = pd.DataFrame(self.X_df, index=self.orig_X_df.index, columns=self.orig_X_df.columns)
        self.input_scaler = scaler
        ## get original by # scaler.inverse_transform(X_train)
        
    def scale_outputs(self, n_quantiles=900):
        """Performs quantile transformation.. as recommended by sklearn """
        index_names = self.y_df.index
        cols_names = self.y_df.columns
        qt = QuantileTransformer(n_quantiles=n_quantiles,output_distribution='normal')
        qt.fit(self.y_df)
        self.output_scaler = qt
        self.y_df = qt.transform(self.y_df)
        self.y_df = pd.DataFrame(self.y_df, index = index_names, columns = cols_names)
        
    def reset_Xy(self):
        self.X_df = self.orig_X_df.copy()
        self.y_df = self.orig_y_df.copy()
        
    def filter_Xy_cols(self, std_thresh=1e-3, verbose=False):
        """Drop features in X_df and y_df that are all 0, or the same number (or have very little std)"""
        if verbose==True:
            print(self.X_df.shape)
        self.X_df = self.X_df[self.X_df.columns[self.X_df.std()>std_thresh]]
        if verbose==True:
            print(self.X_df.shape)
        
        if verbose==True:
            print(self.y_df.shape)
        self.y_df = self.y_df[self.y_df.columns[self.y_df.std()>std_thresh]]
        if verbose==True:
            print(self.y_df.shape)
        self.medium_cols = [x for x in self.y_df.columns if "__medium" in x]
        
    def filter_features(self, n_best=100, reg_type="regression"):
        """Filter the features based on f_regression. Takes average score across columns in y_df
        reg_type either "regression" or "mutual_info_regression"
        """
        if reg_type=="regression":
            kbest_regr = f_regression
        elif reg_type=="mutual_info_regression":
            kbest_regr = mutual_info_regression
        selected_features = [] 
        for label in self.y_df.columns:
            selector = SelectKBest(kbest_regr, k='all')
            selector.fit(self.X_df, self.y_df[label])
            selected_features.append(list(selector.scores_))
            
        score_list = np.mean(selected_features, axis=0)
        kbest_indices = heapq.nlargest(n_best, range(len(score_list)), key=lambda x: score_list[x])
        new_cols = self.X_df.columns[kbest_indices]
        self.X_df = self.X_df[new_cols]
        return new_cols
    
    def plot_pred_single_target(self):
        """ only works for single target variable"""
        X_train, X_test, y_train, y_test = train_test_split(self.X_df, self.y_df, random_state=0)
        self.nn.fit(X_train, y_train)
        y_pred = self.nn.predict(X_test)
        self.nn.score(X_test, y_test)

        f, (ax0) = plt.subplots(1, 1, sharey=True)
        ax0.set_xlabel("Actual")
        ax0.set_ylabel("Predicted")
        # ax0.plot([-5, 0], [min(y_pred), max(y_pred)], "--k")
        ax0.text(
            #0.8*(max(y_test.values)-abs(min(y_test.values)))[0],
            #0.7*(max(y_pred)-abs(min(y_pred))),
            0.8*min(y_test.values),
            0.8*max(y_test.values),
            # -4,4,
            r"$R^2$=%.2f, MAE=%.2f"% (r2_score(y_test, y_pred), median_absolute_error(y_test, y_pred)),
        )
        # ax0.plot([-5, 5], [-5, 5], "--k")
        ax0.plot([min(y_test.values), max(y_test.values)], [min(y_test.values), max(y_test.values)], "--k")
        ax0.scatter(y_test, y_pred)
        return f, ax0
        
    def load_micom_data(self, X_df, y_df):
        self.orig_X_df = X_df.copy()
        self.orig_y_df = y_df.copy()
        self.X_df = X_df.copy()
        self.y_df = y_df.copy()
        self.medium_cols = [x for x in self.y_df.columns if "__medium" in x]