import pandas as pd
import pickle
import itertools
import feature_utils
from tqdm import tqdm
import copy
import argparse
import os
from os import listdir
from os.path import isfile, join
import pandas as pd
from autogluon.tabular import TabularDataset, TabularPredictor
from multiprocessing import Process, Manager, Pool
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from statistics import mean

with open(f'UCI_data_0_10_100.pkl', 'rb') as f:
    xs = pickle.load(f)

datas = []
clone_frag_dict = dict()
k = 10
for i, x in enumerate(xs):
    # print(x)
    # exit()
    data = []
    # data.extend(x['ltc_sim_features'][0:3])
    for sim_mat in x['ml_sim_features']:
        top_k = sim_mat[:k]
        if len(top_k) == 0:
            data.append(0.)
        else:
            data.append(mean(top_k))
    data.append(x['target'])
    datas.append(data)

df = pd.DataFrame(datas,
                  columns=['jaccard_str_row', 'jaccard_str_col', 'jaccard_num_row', 'jaccard_num_col', 'mean_col',
                           'dev_col', 'mean_row', 'dev_row', 'simhash_col', 'simhash_row', 'lev_col', 'lev_row',
                           'textrank_col', 'textrank_row', 'target'])
from sklearn.model_selection import train_test_split

train, test = train_test_split(df, test_size=0.2, random_state=42)
# get pre-trained model from previous step
predictor = TabularPredictor.load('ML_predictor/')
test_target = test[['target']]
test_feature = test[
    ['jaccard_str_row', 'jaccard_str_col', 'jaccard_num_row', 'jaccard_num_col', 'mean_col', 'dev_col', 'mean_row',
     'dev_row', 'simhash_col', 'simhash_row', 'lev_col', 'lev_row', 'textrank_col', 'textrank_row']]
try:
    test_target['clone probability'] = predictor.predict_proba(test_feature)[1]
except:
    pass
df_test_clone = test_target[test_target['clone probability'] > 0.5]
df_test_clone = df_test_clone[df_test_clone['target'] == 1]
print(df_test_clone.info())
print(df_test_clone)
# exit()

def cal_sim_mats(x):
    df_1, df_2 = x.get('df_1'), x.get('df_2')
    sim_type, mat_list = feature_utils.val_sim(df_1, df_2)
    return mat_list


xs_test_clone = [xs[i] for i in df_test_clone.index.tolist()]
mat_list_list = []
from itertools import repeat
with ProcessPoolExecutor(max_workers=48) as executor:
    for r in executor.map(cal_sim_mats, xs_test_clone):
        mat_list_list.append(r)
with open('mat_list.pkl','wb') as f:
    pickle.dump(mat_list_list,f)
with open('mat_list.pkl', 'rb') as f:
    mat_list_list = pickle.load(f)



df_test_clone_mat_dict = dict(zip(df_test_clone.index.tolist(), mat_list_list))
X_train = train[['jaccard_str_row', 'jaccard_str_col', 'jaccard_num_row',
                 'jaccard_num_col', 'mean_col', 'dev_col', 'mean_row', 'dev_row', 'simhash_col', 'simhash_row',
                 'lev_col', 'lev_row', 'textrank_col', 'textrank_row']]
y_train = train[['target']].squeeze()
target_class = 1
negative_class = 0
baseline = X_train[y_train == negative_class].sample(1000, random_state=0)
import shap


class AutogluonWrapper:
    def __init__(self, predictor, feature_names, target_class=None):
        self.ag_model = predictor
        self.feature_names = feature_names
        self.target_class = target_class
        if target_class is None and predictor.problem_type != 'regression':
            print("Since target_class not specified, SHAP will explain predictions for each class")

    def predict_proba(self, X):
        if isinstance(X, pd.Series):
            X = X.values.reshape(1, -1)
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X, columns=self.feature_names)
        preds = self.ag_model.predict_proba(X)
        if predictor.problem_type == "regression" or self.target_class is None:
            return preds
        else:
            return preds[self.target_class]


ag_wrapper = AutogluonWrapper(predictor, X_train.columns, target_class)
explainer = shap.KernelExplainer(ag_wrapper.predict_proba, baseline)
NSHAP_SAMPLES = 200
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import numpy.ma as ma
from shap import LinearExplainer, KernelExplainer, Explanation


def find_column_inclusive(df_1, df_2, index_table_1, index_table_2, df_frac):
    for index in range(df_frac.shape[1]):
        if dict(df_1.iloc[:, index_table_1].value_counts()) == dict(df_frac.iloc[:, index].value_counts()):
            return True
        if dict(df_2.iloc[:, index_table_2].value_counts()) == dict(df_frac.iloc[:, index].value_counts()):
            return True
    return False


def find_row_inclusive(df_1, df_2, index_table_1, index_table_2, df_frac):
    for index in range(df_frac.shape[1]):
        if dict(df_1.iloc[index_table_1].value_counts()) == dict(df_frac.iloc[index].value_counts()):
            return True
        if dict(df_2.iloc[index_table_2].value_counts()) == dict(df_frac.iloc[index].value_counts()):
            return True
    return False


import warnings

warnings.filterwarnings('ignore')
from scipy.linalg import block_diag

total = df_test_clone.shape[0]
shap_hit_count = 0
hit_count = 0
counter = 0
import feature_utils
from numpy import unravel_index

mean_top_ten_hit_lists = []
shap_top_ten_hit_lists = []
for index, row in df_test_clone.iterrows():
    counter += 1
    print(index)
    shap_values_single = explainer.shap_values(test_feature.loc[index], nsamples=NSHAP_SAMPLES)
    df_1, df_2 = xs[index].get('df_1'), xs[index].get('df_2')
    # sim_type,mat_list = feature_utils.val_sim(df_1,df_2)
    ## if pre-cal
    mat_list = df_test_clone_mat_dict.get(index)
    mat_list = [np.clip(mat, 0, 1) for mat in mat_list]
    df_1_str = df_1.select_dtypes(include=[object])
    df_2_str = df_2.select_dtypes(include=[object])
    df_1_num = df_1.select_dtypes(include=[np.number])
    df_2_num = df_2.select_dtypes(include=[np.number])
    shap_values_single = shap_values_single.squeeze()
    for i, mat in enumerate(mat_list):
        if mat.size == 0:
            shap_values_single[i] = 0
    # print(row['file_1'],row['file_2'])
    heatmaps = dict()
    shap_weights = dict()
    heatmaps['str_row'] = [mat_list[0], mat_list[9], mat_list[11], mat_list[13]]
    shap_weights['str_row'] = [shap_values_single[0], shap_values_single[9], shap_values_single[11],
                               shap_values_single[13]]
    heatmaps['str_col'] = [mat_list[1], mat_list[8], mat_list[10], mat_list[12]]
    shap_weights['str_col'] = [shap_values_single[1], shap_values_single[8], shap_values_single[10],
                               shap_values_single[12]]
    heatmaps['num_row'] = [mat_list[2], mat_list[6], mat_list[7]]
    shap_weights['num_row'] = [shap_values_single[2], shap_values_single[6], shap_values_single[7]]
    heatmaps['num_col'] = [mat_list[3], mat_list[4], mat_list[5]]
    shap_weights['num_col'] = [shap_values_single[3], shap_values_single[4], shap_values_single[5]]
    shap_heatmaps = dict()
    for i, heatmap in heatmaps.items():
        weight_np = np.array(shap_weights[i])
        non_zero = [heat * weight_np[i] for i, heat in enumerate(heatmap) if heat.size != 0]
        shap_heatmaps[i] = np.sum(np.array(non_zero), axis=0)
        heatmaps[i] = np.mean(np.array(heatmap), axis=0)

    figure, axes = plt.subplots(1, 2, sharex=True, figsize=(10, 5))


    plt_shap_row_heatmap = shap_heatmaps['str_row'] + shap_heatmaps['num_row']
    plt_shap_col_heatmap = block_diag(shap_heatmaps['str_col'], shap_heatmaps['num_col'])
    min_heat = min(np.min(plt_shap_row_heatmap), np.min(plt_shap_col_heatmap))
    max_heat = max(np.max(plt_shap_row_heatmap), np.max(plt_shap_col_heatmap))
    plt_shap_row_heatmap = (plt_shap_row_heatmap - min_heat) / (max_heat - min_heat)
    plt_shap_col_heatmap = (plt_shap_col_heatmap - min_heat) / (max_heat - min_heat)

    if plt_shap_row_heatmap.size != 1:
        sns.heatmap(ax=axes[1,0],data = plt_shap_row_heatmap,cmap="YlGnBu",vmin=0,vmax=1)
        axes[1,0].title.set_text('row-row')
    # print(plt_shap_col_heatmap.shape)
    if plt_shap_col_heatmap.size != 1:
        sns.heatmap(ax=axes[1,1],data = plt_shap_col_heatmap,cmap="YlGnBu",vmin=0,vmax=1)
        axes[1,1].title.set_text('col-col')
    plt.show()
