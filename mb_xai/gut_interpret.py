import numpy as np
import pandas as pd
import os
from os import listdir, path, makedirs
from collections import Counter
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.feature_selection import SelectKBest, f_classif
from tqdm import tqdm
from scipy.stats import hypergeom
import networkx as nx
from scipy import stats
## Plotting stuff
import matplotlib.pyplot as plt
## mb_xai stuff

from mb_xai import mb_utils


class InterpretPred(object):
    """ Class object holding information flux and abundance predictions
        - Generates plots + other stuff
    """
    def __init__(self):
        self.dir_sim_data = None
        self.SAVE_FIG = True
        
        self.input_type_aucs_long = pd.DataFrame()
        self.imp_feat_abundance_pheno_df = pd.DataFrame()
        self.imp_feat_flux_pheno_df = pd.DataFrame()
        self.flux_pheno_direct_df = pd.DataFrame()
        self.food_matrix_df = pd.DataFrame()
        self.imp_feat_flux_pheno_df_metab = pd.DataFrame()
        self.flux_pheno_direct_df_metab = pd.DataFrame()
        ## computation parameters
        self.X_flux = pd.DataFrame()
        self.y_df = pd.DataFrame() # metabolite signal df for all phenotypes
        self.A_df = pd.DataFrame() # filtered food matrix
        self.food_signal_df = pd.DataFrame() # food signal vector for all phenotypes
        
    def load_pred(self,
                  gut_data, # gut_data object from gut_data.py 
                  DIR_SIM_DATA = "../data/micom-sim-data/",
                  DATA_LOC = '../../../Data/microbiome_xai/',
                  SAVE_ID = "PARALLEL_std-4_noparams_5_75",
                  SAVE_FIG = True,
                  BOOL_FLUX_NOTMEDIUM = False
    ):
        self.gut_data = gut_data
        self.SAVE_FIG = SAVE_FIG
        self.SAVE_ID = SAVE_ID
        self.dir_sim_data = DIR_SIM_DATA
        self.BOOL_FLUX_NOTMEDIUM = BOOL_FLUX_NOTMEDIUM
        self.input_type_aucs_long = pd.read_csv(DIR_SIM_DATA+'ml_performance_%s_fd.csv'%(SAVE_ID), index_col=0)
        self.imp_feat_abundance_pheno_df = pd.read_csv(DIR_SIM_DATA+'imp_feat_abundance_%s_fd.csv'%(SAVE_ID), index_col=0)
        self.imp_feat_flux_pheno_df = pd.read_csv(DIR_SIM_DATA+'imp_feat_flux_%s_fd.csv'%(SAVE_ID), index_col=0)
        self.flux_pheno_direct_df = pd.read_csv(DIR_SIM_DATA+'imp_flux_direction_%s_fd.csv'%(SAVE_ID), index_col=0)
        self.flux_pheno_direct_df.columns = [x.split("_")[1]+"_"+x.split("_")[0] for x in self.flux_pheno_direct_df.columns]
        self.food_matrix_df = pd.read_csv(DATA_LOC+'tables/food_matrix_df.csv',index_col=0)
        
        
        if self.BOOL_FLUX_NOTMEDIUM==False:
            self.imp_feat_flux_pheno_df_metab = mb_utils.get_metab_name_df(self.imp_feat_flux_pheno_df, self.gut_data)
            self.flux_pheno_direct_df_metab = mb_utils.get_metab_name_df(self.flux_pheno_direct_df, self.gut_data)
    
    
    def plot_performance_table(self, NO_DUMMY=True):
        """Saves results as a clean table with mean and std of performance metrics
        Args:
            NO_DUMMY (bool, optional): _description_. Defaults to True.
        """
        SAVE_LOC = self.dir_sim_data+'performance_table_%s_fd.csv'%(self.SAVE_ID)
        self.performance_table = mb_utils.plot_performance_table(self.input_type_aucs_long, SAVE_LOC, NO_DUMMY=NO_DUMMY)
        
        
    def plot_flux_genera_topfeats(self, n_feats=10, METAB_NAME=True):
        """Plot horizontal barplot of top flux and genera features for all phenotypes
        """
        if METAB_NAME==True:
            imp_feat_flux_df = self.imp_feat_flux_pheno_df_metab
        else:
            imp_feat_flux_df = self.imp_feat_flux_pheno_df
            
        # for input_type, input_df in [("flux", imp_feat_flux_pheno_df), ("abundance", imp_feat_abundance_pheno_df)]:
        for input_type, input_df in [("flux", imp_feat_flux_df), ("abundance", self.imp_feat_abundance_pheno_df)]:

            f, ax = plt.subplots(1, len(input_df.columns), figsize=(15, n_feats/2)) # 5 works well for n_feats=10
            for i, col in enumerate(input_df.columns):
                top_pos_df = input_df[col].sort_values(ascending=False)[:n_feats][::-1]
                top_neg_df = input_df[col].sort_values(ascending=True)[:n_feats][::-1]
                top_df = pd.DataFrame(pd.concat([top_pos_df, top_neg_df]))
                top_df['positive'] = top_df[col] > 0
                top_df = top_df.sort_values(by=col,ascending=False)[::-1]
                top_df[col].plot(
                    kind="barh", color=top_df.positive.map({True: 'skyblue', False: 'lightcoral'}), ax=ax[i])
                # abs(input_df)[col].sort_values(ascending=False)[:5][::-1].plot(kind="barh", ax=ax[i])
                ax[i].set_xlabel("Avg LogReg coefficient")
                ax[i].set_title("%s"%(col))

            if self.SAVE_FIG == True:
                f.tight_layout()
                f.savefig(self.dir_sim_data+"figures/"+"LogReg_%s_feature_barh_fd_%s.svg"%(input_type, self.SAVE_ID))
                f.savefig(self.dir_sim_data+"figures/"+"LogReg_%s_feature_barh_fd_%s.png"%(input_type, self.SAVE_ID))
                
    
    def infer_foods(self, 
                    bool_concentrations=True, bool_direct_flux=True, bool_consumption=False, 
                    normalize=True, SAVE_ID_ALPHA=4, PVAL_FILT=None):
        """Performs lasso to identify a food vector representing the key metabolite vector.
        Args:
            bool_concentrations (bool, optional): whether to look at quantitative or binary metabolite concentrations in food. Defaults to True.
            bool_direct_flux (bool, optional): whether to correct for direction of flux. Defaults to True.
            bool_consumption (bool, optional): whether to filter for metabolites that are consumed. Defaults to False.
            normalize (bool, optional): _description_. Defaults to True.
            SAVE_ID_ALPHA (int, optional): _description_. Defaults to 4.
        """
        self.bool_concentrations = bool_concentrations
        self.bool_direct_flux = bool_direct_flux
        self.bool_consumption = bool_consumption
        self.SAVE_ID_ALPHA = SAVE_ID_ALPHA
        
        SAVE_LOC_LASSO = self.dir_sim_data+"figures/"+"food_lasso_scatter_fd_%s_conc-%s_fluxdirect-%s_%s.svg"%(
            str(SAVE_ID_ALPHA), str(bool_concentrations), str(bool_direct_flux), self.SAVE_ID)

        self.y_df, self.A_df = mb_utils.init_flux_food_df(
            self.imp_feat_flux_pheno_df.copy(), self.flux_pheno_direct_df.copy(), self.food_matrix_df, self.gut_data.X_flux_consumed_cols, 
            bool_concentrations=bool_concentrations, bool_direct_flux=bool_direct_flux, bool_consumption=bool_consumption)

        self.food_signal_df = mb_utils.food_lasso(
            self.y_df, self.A_df, SAVE_LOC_LASSO, SAVE_ID_ALPHA=SAVE_ID_ALPHA,SAVE_FIG=self.SAVE_FIG,
            normalize=normalize, PVAL_FILT=PVAL_FILT)
        
    def plot_food_topfeats(self, n_feats=10):
        """Plot horizontal barplot of top food features for all phenotypes
        """
        input_df = self.food_signal_df.copy()
        input_df.index = input_df.index.map(lambda x: x[:30] if len(x) > 25 else x)
        f, ax = plt.subplots(1, len(input_df.columns), figsize=(12, n_feats/2)) # 5 works well for n_feats=10
        # SAVE_ID="a3"
        for i, col in enumerate(input_df.columns):
            top_pos_df = input_df[col].sort_values(ascending=False)[:n_feats][::-1]
            top_neg_df = input_df[col].sort_values(ascending=True)[:n_feats][::-1]
            top_df = pd.DataFrame(pd.concat([top_pos_df, top_neg_df]))
            top_df['positive'] = top_df[col] > 0
            top_df = top_df.sort_values(by=col,ascending=False)[::-1]
            top_df[col].plot(
                kind="barh", color=top_df.positive.map({True: 'skyblue', False: 'lightcoral'}), ax=ax[i])
            # abs(input_df)[col].sort_values(ascending=False)[:5][::-1].plot(kind="barh", ax=ax[i])
            ax[i].set_xlabel("Lasso coefficient")
            ax[i].set_title("%s"%(col))

        if self.SAVE_FIG == True:
            f.tight_layout()
            f.savefig(self.dir_sim_data+"figures/"+"foods_features_%s_barh_fd_conc-%s_fluxdirect-%s_%s.png"%(str(self.SAVE_ID_ALPHA), str(self.bool_concentrations), str(self.bool_direct_flux),self.SAVE_ID))
            f.savefig(self.dir_sim_data+"figures/"+"foods_features_%s_barh_fd_conc-%s_fluxdirect-%s_%s.svg"%(str(self.SAVE_ID_ALPHA), str(self.bool_concentrations), str(self.bool_direct_flux),self.SAVE_ID))
            
            
    def plot_pheno_topfeats(self, pheno="vegan", n_feats=8, METAB_NAME=True, BOOL_DIRECT=False, PVAL_THRESH=0.05):
        """Plot horizontal barplot of top genera, fluxes, and foods for a specified phenotype
        Phenotypes could either be vegan, ibs, ibd, or t2d
        """
        if METAB_NAME==True:
            imp_feat_flux_df = self.imp_feat_flux_pheno_df_metab
        else:
            imp_feat_flux_df = self.imp_feat_flux_pheno_df
            
        if BOOL_DIRECT==True:
            imp_feat_flux_df = self.flux_pheno_direct_df_metab
            
        f, ax = plt.subplots(1, 3, figsize=(12, n_feats/2)) # 5 works well for n_feats=10
        for i, (input_type, input_df) in enumerate([
            ("abundance", self.imp_feat_abundance_pheno_df), ("flux", imp_feat_flux_df), ("diet", self.food_signal_df)
            # ("abundance", imp_feat_abundance_pheno_df), ("flux", imp_feat_flux_pheno_df_metab), ("diet", food_signal_df)
            ]):
            col = pheno+"_"+input_type
            if input_type=="diet":
                input_df.index = input_df.index.map(lambda x: x[:30] if len(x) > 25 else x)
                col = pheno+"_"+"flux"
            top_pos_df = input_df[col].sort_values(ascending=False)[:n_feats][::-1]
            top_neg_df = input_df[col].sort_values(ascending=True)[:n_feats][::-1]
            top_df = pd.DataFrame(pd.concat([top_pos_df, top_neg_df]))
            top_df['positive'] = top_df[col] > 0
            top_df = top_df.sort_values(by=col,ascending=False)[::-1]
            top_df[col].plot(
                kind="barh", color=top_df.positive.map({True: 'skyblue', False: 'lightcoral'}), ax=ax[i])
            # abs(input_df)[col].sort_values(ascending=False)[:5][::-1].plot(kind="barh", ax=ax[i])
            ax[i].set_xlabel("Lasso coefficient")
            if input_type=="diet":
                ax[i].set_title("%s"%(pheno+"_"+input_type))
            else:
                ax[i].set_title("%s"%(col))
                
            p_vals = mb_utils.get_pvalues(input_df[col], sig_cutoff=PVAL_THRESH)
            for k, v in enumerate(top_df[col]):
                if top_df[col].index[k] in p_vals.index:
                    ax[i].text(v*1.1, k + -0.4, "*", color='black', fontweight='bold', size=15)

        if self.SAVE_FIG == True:
            f.tight_layout()
            f.savefig(self.dir_sim_data+"figures/"+"key_feats_barh_%s_%s.png"%(pheno, self.SAVE_ID))
            f.savefig(self.dir_sim_data+"figures/"+"key_feats_barh_%s_%s.svg"%(pheno, self.SAVE_ID))
        return f, ax
            
            
            
            
#### Network Visualization helper functions
def get_node_colors(G, imp_feat_flux_pheno_df, imp_feat_abundance_pheno_df, pheno="vegan",cutoff_val = 0.05):
    color_map = []
    node_2color_dict = {}
    for node in G:
        # print(node)
        color_id=None
        for input_type, input_df in [("flux", imp_feat_flux_pheno_df), ("abundance", imp_feat_abundance_pheno_df)]:
            if node in input_df[pheno+"_"+input_type].index:
                if input_df[pheno+"_"+input_type].loc[node]>cutoff_val :
                    color_id = 'skyblue'
                elif input_df[pheno+"_"+input_type].loc[node]<-cutoff_val :
                    color_id = 'lightcoral'
                else:
                    color_id = "grey"
                color_map.append(color_id)
                node_2color_dict.update({node: color_id})
        if color_id==None:
            color_map.append("grey")
            node_2color_dict.update({node: "grey"})
            
    return node_2color_dict, color_map


def genera_flux_corr(medium_react, gut_data, X_flux_notmedium,pheno="vegan",SAMPLE_NUM=10000,scale=True):
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
    X, y = mb_utils.match_Xy_df(X_flux_notmedium, y_df)
                
    if len(react_ids)==1:
        print("yes:",X[react_ids[0]].loc[y[y==1].index].mean())
        print("no:",X[react_ids[0]].loc[y[y==0].index].mean())
    else:
        yes_series = X[react_ids].loc[y[y==1].index].mean().sort_values()
        yes_series.name = "yes"
        no_series = X[react_ids].loc[y[y==0].index].mean().sort_values()
        no_series.name = "no"
        yes_no_df = pd.concat([yes_series, no_series],axis=1)
        print(yes_no_df)
        # print("yes:",X[react_ids].loc[y[y==1].index].mean().sort_values())
        # print("no:",X[react_ids].loc[y[y==0].index].mean().sort_values())

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

def plot_nx_genera_flux(gut_data,SAVE_ID, df_pheno, imp_feat_flux, imp_feat_abundance_pheno_df, pheno="vegan", cutoff_val = 0.2, SAVE_FIG=False):
    f, ax = plt.subplots(1,1,figsize=(5,4))
    G = nx.Graph()
    left_nodes = list(set(df_pheno.stack().reset_index()["level_0"].values))
    right_nodes = list(set(df_pheno.stack().reset_index()["level_1"].values))
    max_node_height = max(len(left_nodes), len(right_nodes))
    G.add_nodes_from(left_nodes, bipartite=0) # Add the node attribute "bipartite"
    G.add_nodes_from(right_nodes, bipartite=1)
    # G.add_nodes_from(right_nodes, bipartite=2)
    # B.add_edges_from([(1,'a'), (1,'b'), (2,'b'), (2,'c'), (3,'c'), (4,'a')])
    # B.add_edges_from(
    #     [(idx[0], idx[1]) for idx, row in df_pheno.stack().iteritems()])
    G.add_weighted_edges_from(
        [(idx[0], idx[1], row) for idx, row in df_pheno.stack().iteritems()], 
        weight='weight')
    # G.add_weighted_edges_from(
    #     [(idx[1], idx[0], row) for idx, row in df_pheno.stack().iteritems()], 
    #     weight='weight')

    edges,weights = zip(*nx.get_edge_attributes(G,'weight').items())
    edge_colors = ["purple" if x >0 else "orange" for x in weights]
    # Separate by group
    # print(B.edges(data=True))

    pos = {node:[0, i-len(left_nodes)/2] for i,node in enumerate(left_nodes)}
    pos.update({node:[1, i-len(right_nodes)/2] for i,node in enumerate(right_nodes)})
    # pos.update({node:[2, i-len(right_nodes)/2] for i,node in enumerate(left_nodes)})
    node_2color_dict, color_map = get_node_colors(
        G, imp_feat_flux, imp_feat_abundance_pheno_df, pheno=pheno,cutoff_val =cutoff_val)
    print(node_2color_dict)
    nx.draw(
        G, pos, with_labels=False, 
        node_color=color_map, 
        edge_color=edge_colors,
        width=1.0,node_size=300, arrows=True)
    for p in pos:  # raise text positions
        pos[p][1] += 0.25
        
    # for edge in G.edges(data=True):
    #     w = edge[2]['weight']
    #     nx.draw_networkx_edges(G, pos, edgelist=[(edge[0],edge[1])], arrowsize=100)
    # plt.show()
    nx.draw_networkx_labels(G, pos)

    #plt.tight_layout()
    f.tight_layout()
    plt.margins(x=0.4,y=0.4)
    #plt.show()
    if SAVE_FIG == SAVE_FIG:
        f.savefig(gut_data.dir_sim_data+"figures/"+"genus_flux_networkx_%s.svg"%(SAVE_ID))
        f.savefig(gut_data.dir_sim_data+"figures/"+"genus_flux_networkx_%s.png"%(SAVE_ID))