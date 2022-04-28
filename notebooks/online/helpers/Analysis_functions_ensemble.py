#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import csv
import matplotlib.pyplot  as plt
from sklearn import tree
from collections import Counter
import sklearn.model_selection
from sklearn.preprocessing import MinMaxScaler, PolynomialFeatures
from scipy import stats
from sklearn.model_selection import cross_val_predict, train_test_split, cross_val_score, KFold, cross_validate, GridSearchCV
from sklearn.svm import LinearSVC, SVC
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.metrics import confusion_matrix, precision_recall_curve, make_scorer, accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.tree import DecisionTreeClassifier
from numpy import mean, std
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.neighbors import KNeighborsClassifier
import seaborn as sns
import warnings
from sklearn.model_selection import RandomizedSearchCV
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from pathlib import Path
import xgboost 
from sklearn.inspection import permutation_importance
from sklearn.utils import resample
from imblearn.under_sampling import RandomUnderSampler
get_ipython().run_line_magic('matplotlib', 'inline')

from graphviz import Source
import pydot
pd.set_option('display.max_colwidth', None)

warnings.filterwarnings('ignore')


# In[2]:


#target_names = ['class drop', 'class stay']

precision_stay_final = []
precision_drop_final = []
recall_stay_final = []
recall_drop_final = []
recall_general_final = []
accuracy_final = []
class_imbalance_final = []
clf_list = []

def add_to_final_table(X_train, y_train, clf, names):
   
    y_pred = cross_val_predict(estimator=clf, X=X_train, y=y_train, cv=5)
    #y_pred = clf.predict(X_test)
    eval_df = classification_report(y_train, y_pred, digits=3, target_names=names, output_dict=True)
    eval_df = pd.DataFrame(eval_df).transpose()
    
    precision_stay = eval_df['precision'][1]
    precision_drop = eval_df['precision'][0]
    recall_stay = eval_df['recall'][1]
    recall_drop = eval_df['recall'][0]
    recall_general = eval_df['recall'][3]
    accuracy = eval_df['precision'][2]
    class_imbalance = eval_df['support'][0] / eval_df['support'][1]
    precision_stay_final.append(precision_stay)
    precision_drop_final.append(precision_drop)
    recall_stay_final.append(recall_stay)
    recall_drop_final.append(recall_drop)
    recall_general_final.append(recall_general)
    accuracy_final.append(accuracy)
    class_imbalance_final.append(class_imbalance)
    clf_list.append(clf)
    
def XGBoost_to_final_table(y_pred, y_train, names):
    eval_df = classification_report(y_train, y_pred, digits=3, target_names=names, output_dict=True)
    eval_df = pd.DataFrame(eval_df).transpose()
    
    precision_stay = eval_df['precision'][1]
    precision_drop = eval_df['precision'][0]
    recall_stay = eval_df['recall'][1]
    recall_drop = eval_df['recall'][0]
    recall_general = eval_df['recall'][3]
    accuracy = eval_df['precision'][2]
    class_imbalance = eval_df['support'][0] / eval_df['support'][1]
    precision_stay_final.append(precision_stay)
    precision_drop_final.append(precision_drop)
    recall_stay_final.append(recall_stay)
    recall_drop_final.append(recall_drop)
    recall_general_final.append(recall_general)
    accuracy_final.append(accuracy)
    class_imbalance_final.append(class_imbalance)
    clf_list.append('XGBoost')


# In[3]:


# checking on test set
precision_stay_test = []
precision_drop_test = []
recall_stay_test = []
recall_drop_test = []
recall_general_test = []
accuracy_test = []
class_imbalance_test = []
clf_test = []

def add_to_final_table_test(X_test, y_test, clf, names):
    #y_pred = cross_val_predict(estimator=clf, X=X_train, y=y_train, cv=5)
    y_pred = clf.predict(X_test)
    eval_df = classification_report(y_test, y_pred, digits=3, target_names=names, output_dict=True)
    eval_df = pd.DataFrame(eval_df).transpose()
    
    precision_stay = eval_df['precision'][1]
    precision_drop = eval_df['precision'][0]
    recall_stay = eval_df['recall'][1]
    recall_drop = eval_df['recall'][0]
    recall_general = eval_df['recall'][3]
    accuracy = eval_df['precision'][2]
    class_imbalance = eval_df['support'][0] / eval_df['support'][1]
    precision_stay_test.append(precision_stay)
    precision_drop_test.append(precision_drop)
    recall_stay_test.append(recall_stay)
    recall_drop_test.append(recall_drop)
    recall_general_test.append(recall_general)
    accuracy_test.append(accuracy)
    class_imbalance_test.append(class_imbalance)
    clf_test.append(clf)


# In[4]:


# cutting unnessesary leaves
# https://stackoverflow.com/questions/51397109/prune-unnecessary-leaves-in-sklearn-decisiontreeclassifier
from sklearn.tree._tree import TREE_LEAF

def is_leaf(inner_tree, index):
    # Check whether node is leaf node
    return (inner_tree.children_left[index] == TREE_LEAF and 
            inner_tree.children_right[index] == TREE_LEAF)

def prune_index(inner_tree, decisions, index=0):
    # Start pruning from the bottom - if we start from the top, we might miss
    # nodes that become leaves during pruning.
    # Do not use this directly - use prune_duplicate_leaves instead.
    if not is_leaf(inner_tree, inner_tree.children_left[index]):
        prune_index(inner_tree, decisions, inner_tree.children_left[index])
    if not is_leaf(inner_tree, inner_tree.children_right[index]):
        prune_index(inner_tree, decisions, inner_tree.children_right[index])

    # Prune children if both children are leaves now and make the same decision:     
    if (is_leaf(inner_tree, inner_tree.children_left[index]) and
        is_leaf(inner_tree, inner_tree.children_right[index]) and
        (decisions[index] == decisions[inner_tree.children_left[index]]) and 
        (decisions[index] == decisions[inner_tree.children_right[index]])):
        # turn node into a leaf by "unlinking" its children
        inner_tree.children_left[index] = TREE_LEAF
        inner_tree.children_right[index] = TREE_LEAF
        ##print("Pruned {}".format(index))

def prune_duplicate_leaves(mdl):
    # Remove leaves if both 
    decisions = mdl.tree_.value.argmax(axis=2).flatten().tolist() # Decision for each node
    prune_index(mdl.tree_, decisions)


# defining all the models with default and tuned parameters, adding results to final table

# In[5]:


#########################################################
# DECISION TREE #########################################
#########################################################
"""   
# trying to reduce complexity (cost complexity pruning) and find optimal parameters using CV:
def decision_tree_ccp_alpha(X_train, y_train, X_test, y_test):

    clf = DecisionTreeClassifier(class_weight="balanced")
    clf.fit(X_train, y_train)
    path = clf.cost_complexity_pruning_path(X_train, y_train)
    ccp_alphas = path['ccp_alphas']
    alpha_loop_values = []
    for alpha in ccp_alphas[-20:]:
        clf_dt = DecisionTreeClassifier(ccp_alpha=alpha)
        scores = cross_val_score(clf_dt, X_train, y_train, cv=5)
        alpha_loop_values.append([alpha, np.mean(scores), np.std(scores)])

    alpha_results = pd.DataFrame(alpha_loop_values, columns=['alpha', 'mean_accuracy', 'std'])
    #alpha_results.plot(x='alpha', y='mean_accuracy', yerr='std', marker='o', figsize=(15, 4))
    alpha_results = alpha_results.sort_values(by=['mean_accuracy'], ascending=False)
    max_result = alpha_results[alpha_results['mean_accuracy']==alpha_results['mean_accuracy'].max()]['alpha']
    opt_alpha = max_result.iloc[0]
    clf = DecisionTreeClassifier(ccp_alpha=opt_alpha)
    clf.fit(X_train, y_train)
    add_to_final_table(X_train, y_train, clf)
    add_to_final_table_test(X_test, y_test, clf)
    
"""
    
# DT    
def decision_tree_250(X_train, y_train, X_test, y_test, names):
    tree_params = {'min_samples_leaf': [20, 50, 100, 250]}
    clf = GridSearchCV(DecisionTreeClassifier(max_depth=4, class_weight="balanced"), tree_params, cv=5)
    clf.fit(X_train, y_train)
    print(clf.best_estimator_)
    #clf = DecisionTreeClassifier(clf.best_estimator_)
    #clf.fit(X_train, y_train)
    prune_duplicate_leaves(clf.best_estimator_)
    
    add_to_final_table(X_train, y_train, clf, names)
    add_to_final_table_test(X_test, y_test, clf, names)
    #print('DecisionTreeClassifier')
    print(' ')
    
    X_train.feature_names = X_train.columns
    plt.figure(figsize=(45, 10))
    _ = tree.plot_tree(clf.best_estimator_, filled=True, feature_names=X_train.feature_names, fontsize=12, precision=7, class_names=names)
    plt.show()
    
                
#########################################################
# SVM (LinearSVC) #######################################
#########################################################
    
# default parameters 
def SVM_default(X_train, y_train, X_test, y_test, names):
        
    clf = LinearSVC(max_iter=10000, class_weight="balanced")
    clf.fit(X_train, y_train)
    add_to_final_table(X_train, y_train, clf, names)
    add_to_final_table_test(X_test, y_test, clf, names)
    
# C=0.01
def SVM_C(X_train, y_train, X_test, y_test, names):
    
    clf = LinearSVC(C=0.01, max_iter=100000, class_weight="balanced")
    clf.fit(X_train, y_train)
    add_to_final_table(X_train, y_train, clf, names)
    add_to_final_table_test(X_test, y_test, clf, names)
    
#########################################################
# RBF SVM ###############################################
#########################################################
    
# default parameters    
def RBF_SVM(X_train, y_train, X_test, y_test, names):
    
    clf = SVC(class_weight="balanced")
    clf.fit(X_train, y_train)
    add_to_final_table(X_train, y_train, clf, names)
    add_to_final_table_test(X_test, y_test, clf, names)
    
#########################################################
# LOGISTIC REGRESSION ###################################
#########################################################
    
# default parameters    
def logistic_regression(X_train, y_train, X_test, y_test, names):
    
    clf = LogisticRegressionCV(cv=5, max_iter=10000, class_weight="balanced").fit(X_train, y_train)
    add_to_final_table(X_train, y_train, clf, names)
    add_to_final_table_test(X_test, y_test, clf, names)
    print('The most important features and its coefficients obtained by logistic regression:')
    feature_coef = pd.DataFrame(clf.coef_)
    feature_coef.columns = X_train.columns
    feature_coef.index = ['coef']
    feature_coef = feature_coef.T
    feature_coef_abs = abs(feature_coef['coef'])
    feature_coef_abs = feature_coef_abs.sort_values(ascending=False)
    feature_coef_abs_20 = feature_coef_abs[:20]
    most_important_features = feature_coef_abs_20.index
    #most_important_features_coeffs = feature_coef.loc[most_important_features]
    
    features = []
    coeffs = []
    #quality_features = df.filter(like='quality', axis=1).columns
    knowledge_features = X_train.loc[:, X_train.columns.isin(['voting_age_awareness_w1',
                                                    'KNOWLEDGE_PARLIAMENTARY_THRESHOLD_w1',
                                                    'know_politicians_ratio',
                                                    'whether_dropped_before',
                                                    'lr_placement_correct'])].columns
    joined_list = [*most_important_features, *knowledge_features] #*quality_features,
    for i in joined_list:
        coeff = feature_coef.T[i][0].round(3)
        print(i, ': ', coeff)

        
#########################################################
# XGBOOST ###############################################
#########################################################

# default parameters
def XGBoost_default(X_train, y_train, X_test, y_test, names):    
        
    model = xgboost.XGBClassifier(verbosity=0, scale_pos_weight=Counter(y_train)[0]/Counter(y_train)[1])
    y_pred = cross_val_predict(estimator=model, X=X_train, y=y_train, cv=5)
    XGBoost_to_final_table(y_pred, y_train, names)

# randomized search
def XGBoost_RS(X_train, y_train, X_test, y_test, names):
    model = xgboost.XGBClassifier(verbosity=0, scale_pos_weight=Counter(y_train)[0]/Counter(y_train)[1])
    param = {
     "eta"    : [0.05, 0.10, 0.15, 0.20, 0.25, 0.30] ,
     "max_depth"        : [3, 4, 5, 6, 10, 50],
     "min_child_weight" : [1, 3, 5, 7],
     "gamma"            : [0.0, 0.1, 0.2, 0.3, 0.4],
     "colsample_bytree" : [0.3, 0.4, 0.5, 0.7],
     'verbosity': [0]
     }
    XGB_random = RandomizedSearchCV(estimator = model, 
                           param_distributions = param, n_iter = 50, 
                           cv = 3)
    XGB_random.fit(X_train, y_train)
    y_pred = cross_val_predict(estimator=XGB_random, X=X_train, y=y_train, cv=5)
    XGBoost_to_final_table(y_pred, y_train, names)
    
#########################################################
# RANDOM FOREST #########################################
#########################################################

# default parameters
def RF_default(X_train, y_train, X_test, y_test, names):    
    
    clf = RandomForestClassifier(class_weight="balanced")
    clf.fit(X_train, y_train)
    add_to_final_table(X_train, y_train, clf, names)
    """
    sorted_idx = clf.feature_importances_.argsort()[:10]
    plt.barh(X_train.columns[sorted_idx], clf.feature_importances_[sorted_idx])
    plt.rcParams["figure.figsize"] = (15, 10)
    plt.rcParams.update({'font.size': 14})
    
    importances = clf.feature_importances_
    forest_importances = pd.Series(importances, index=X_train.columns)
    std = np.std([
        tree.feature_importances_ for tree in clf.estimators_], axis=0)
    fig, ax = plt.subplots()
    forest_importances[:10].plot.bar(yerr=std[:10], ax=ax)
    ax.set_title("Feature importance - MDI")
    ax.set_ylabel("Mean decrease in impurity (MDI)")
    fig.tight_layout()
    
    result = permutation_importance(
        clf, X_test, y_test, n_repeats=10, random_state=42, n_jobs=2)
    forest_importances = pd.Series(result.importances_mean, index=X_train.columns)
    fig, ax = plt.subplots()
    forest_importances[:10].plot.bar(yerr=result.importances_std[:10], ax=ax)
    ax.set_title("Feature importance after permutation")
    ax.set_ylabel("Decrease of error")
    fig.tight_layout()
    plt.rcParams.update({'font.size': 14})
    plt.show()
    """
# randomized search
def RF_random_search(X_train, y_train, X_test, y_test, names): 
    
    clf = RandomForestClassifier(class_weight="balanced")
    random_grid = {'bootstrap': [True, False],
     'max_depth': [50, 100, 200],
     'min_samples_leaf': [5, 10, 20],
     'min_samples_split': [20, 40],
     'n_estimators': [500, 2000, 4000]}
    clf = RandomizedSearchCV(estimator = clf, param_distributions = random_grid, n_iter = 50, cv = 3, verbose=2, random_state=42, n_jobs = -1)
    # Fit the random search model
    clf.fit(X_train, y_train)
    print('Random_forest_best: ', clf.best_params_)
    add_to_final_table(X_train, y_train, clf, names)
    """
    sorted_idx = clf.feature_importances_.argsort()[:10]
    plt.barh(X_train.columns[sorted_idx], clf.feature_importances_[sorted_idx])
    plt.rcParams["figure.figsize"] = (15, 10)
    plt.rcParams.update({'font.size': 14})
    
    importances = clf.feature_importances_
    forest_importances = pd.Series(importances, index=X_train.columns)
    std = np.std([
        tree.feature_importances_ for tree in clf.estimators_], axis=0)
    fig, ax = plt.subplots()
    forest_importances[:10].plot.bar(yerr=std[:10], ax=ax)
    ax.set_title("Feature importance - MDI")
    ax.set_ylabel("Mean decrease in impurity (MDI)")
    fig.tight_layout()
    
    
    result = permutation_importance(
        clf, X_test, y_test, n_repeats=10, random_state=42, n_jobs=2)
    forest_importances = pd.Series(result.importances_mean, index=X_train.columns)
    fig, ax = plt.subplots()
    forest_importances[:10].plot.bar(yerr=result.importances_std[:10], ax=ax)
    ax.set_title("Feature importance after permutation")
    ax.set_ylabel("Decrease of error")
    fig.tight_layout()
    plt.rcParams.update({'font.size': 14})
    plt.show()
    """


# In[6]:


def concat_df(df, political_data):
    personal_data = df.drop(['panelpat'], axis=1)
    df = pd.concat([personal_data, political_data], axis=1)
    return df

def scale_train(X_train, X_test):
    cols = X_train.columns
    scaler = MinMaxScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_train = pd.DataFrame(X_train)
    X_train.columns = cols
    X_test = scaler.transform(X_test)
    X_test = pd.DataFrame(X_test)
    X_test.columns = cols
    return X_train, X_test


# In[7]:


def analysis_all_models(X_train, y_train, X_test, y_test, names):
    XGBoost_RS(X_train, y_train, X_test, y_test, names)
    XGBoost_default(X_train, y_train, X_test, y_test, names)
    RF_default(X_train, y_train, X_test, y_test, names)
    RF_random_search(X_train, y_train, X_test, y_test, names)
   # decision_tree_ccp_alpha(X_train, y_train, X_test, y_test)
    #decision_tree_250(X_train, y_train, X_test, y_test, names)
    
    #X_train, X_test = scale_train(X_train, X_test)
    #SVM_default(X_train, y_train, X_test, y_test, names)
    #SVM_C(X_train, y_train, X_test, y_test, names)
    #RBF_SVM(X_train, y_train, X_test, y_test, names)
    #logistic_regression(X_train, y_train, X_test, y_test, names)


# In[9]:


get_ipython().system('jupyter nbconvert --to script heavy_Analysis_functions.ipynb')


# In[ ]:




