import lightgbm as lgb
import numpy as np 
import pandas as pd
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import graphviz





if __name__=='__main__':
    df = pd.read_csv('data.csv')
    X = df.drop(['label','query','term'], axis=1)
    y = df.label
    group=np.loadtxt('./group.txt')
    train_data = lgb.Dataset(X, label=y, group=group,free_raw_data=False)
    params = {
    'task' : 'train', 
    'boosting_type': 'gbdt',
    'objective': 'lambdarank',
    'metric': 'ndcg',
    'max_position': 10,
    'metric_freq': 1,
    'train_metric':True,
    'ndcg_at':[2],
    'max_bin':255,
    'num_iterations': 100,
    'learning_rate':0.01,
    'num_leaves': 31,
    'tree_learner': 'serial',
    #'max_depth': 1,
    'verbose':2
    }
    categorical_feature=[0,1]
    gbm=lgb.train(params,
              train_data,
              valid_sets=train_data,
              categorical_feature=categorical_feature)
    gbm.save_model('model.md')
