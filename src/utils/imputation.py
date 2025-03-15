import pandas as pd
import numpy as np
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer, KNNImputer, SimpleImputer
from missforest import MissForest

# For MICE, use IterativeImputer
def apply_mice(masked_data, estimator=None, random_state=0):
    imputer = IterativeImputer(estimator=estimator, random_state=random_state)
    return imputer.fit_transform(masked_data)

# MissForest implementation from https://github.com/yuenshingyan/MissForest/tree/main
def apply_missforest(data, classifier=None, regressor=None):
    mf = MissForest(clf=classifier, rgr=regressor)
    return mf.fit_transform(data)


def apply_knn_imputer(masked_data, n_neighbors):
    knn_imputer = KNNImputer(n_neighbors=n_neighbors)
    return knn_imputer.fit_transform(masked_data)

def apply_mean_imputation(masked_data):
    imputer = SimpleImputer(strategy='mean')
    return imputer.fit_transform(masked_data)
