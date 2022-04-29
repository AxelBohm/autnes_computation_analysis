#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pydot
from graphviz import Source
import pandas as pd
import numpy as np
import csv
import matplotlib.pyplot as plt
from IPython.display import display_html
from sklearn import tree
from collections import Counter
import sklearn.model_selection
from sklearn.preprocessing import MinMaxScaler, PolynomialFeatures
from scipy import stats
from sklearn.model_selection import cross_val_predict, train_test_split, cross_val_score, KFold, cross_validate, GridSearchCV
from sklearn.svm import LinearSVC, SVC
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.metrics import cohen_kappa_score, roc_auc_score, confusion_matrix, precision_recall_curve, make_scorer, accuracy_score, precision_score, recall_score, f1_score, classification_report #brier_score_loss
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
from sklearn.utils import resample
from imblearn.under_sampling import RandomUnderSampler
get_ipython().run_line_magic('matplotlib', 'inline')

pd.set_option('display.max_colwidth', None)

warnings.filterwarnings('ignore')


# In[ ]:


#target_names = ['class drop', 'class stay']

precision_stay_train = []
precision_drop_train = []
recall_stay_train = []
recall_drop_train = []
recall_general_train = []
accuracy_train = []
rocauc_train = []
cohen_kappa_train = []
#brier_loss_train = []
class_imbalance_train = []
#clf_list = []
algorithm_train = []

def add_to_final_table(X_train, y_train, algorithm, clf, names):
   
    y_pred = cross_val_predict(estimator=clf, X=X_train, y=y_train)
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
    if algorithm == 'SVM_def' or algorithm == 'SVM_C':
        rocauc = roc_auc_score(y_train, clf.decision_function(X_train))
    else:
        rocauc = roc_auc_score(y_train, clf.predict_proba(X_train)[:, 1])
    cohen_kappa = cohen_kappa_score(y_train, y_pred)
    precision_stay_train.append(precision_stay)
    precision_drop_train.append(precision_drop)
    recall_stay_train.append(recall_stay)
    recall_drop_train.append(recall_drop)
    recall_general_train.append(recall_general)
    accuracy_train.append(accuracy)
    rocauc_train.append(rocauc)
    cohen_kappa_train.append(cohen_kappa)
    #brier_loss_train.append(brier_loss)
    class_imbalance_train.append(class_imbalance)
    algorithm_train.append(algorithm)
    
"""    
def XGBoost_to_final_table(y_pred, y_train):
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
    algorithm_list.append('XGBoost')
"""


# In[ ]:


# checking on test set
precision_stay_test = []
precision_drop_test = []
recall_stay_test = []
recall_drop_test = []
recall_general_test = []
accuracy_test = []
rocauc_test = []
cohen_kappa_test = []
class_imbalance_test = []
algorithm_test = []

def add_to_final_table_test(X_test, y_test, algorithm, clf, names):
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
    if algorithm == 'SVM_def' or algorithm == 'SVM_C':
        rocauc = roc_auc_score(y_test, clf.decision_function(X_test))
    else:
        rocauc = roc_auc_score(y_test, clf.predict_proba(X_test)[:, 1]) 
    cohen_kappa = cohen_kappa_score(y_test, y_pred)
    precision_stay_test.append(precision_stay)
    precision_drop_test.append(precision_drop)
    recall_stay_test.append(recall_stay)
    recall_drop_test.append(recall_drop)
    recall_general_test.append(recall_general)
    accuracy_test.append(accuracy)
    rocauc_test.append(rocauc)
    cohen_kappa_test.append(cohen_kappa)
    class_imbalance_test.append(class_imbalance)
    algorithm_test.append(algorithm)


# In[ ]:


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
    decisions = mdl.tree_.value.argmax(axis=2).flatten().tolist() # Decision for each node
    prune_index(mdl.tree_, decisions)


# defining all the models with default and tuned parameters, adding results to final table

# In[ ]:


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


def decision_tree(X_train, y_train, X_test, y_test, names):
    weight_1 = Counter(y_train)[0]/y_train.shape[0]*0.9
    weight_0 = 1 - weight_1
    tree_params = {'min_samples_leaf': [20, 50, 100, 250]}
    clf = GridSearchCV(DecisionTreeClassifier(max_depth=4, class_weight={
                       0: weight_0, 1: weight_1}), tree_params)
    clf.fit(X_train, y_train)
    print(clf.best_estimator_)
    #clf = DecisionTreeClassifier(clf.best_estimator_)
    #clf.fit(X_train, y_train)
    prune_duplicate_leaves(clf.best_estimator_)

    add_to_final_table(X_train, y_train, 'DT', clf, names)
    add_to_final_table_test(X_test, y_test, 'DT', clf, names)
    # print('DecisionTreeClassifier')
    print(' ')

    X_train.feature_names = X_train.columns
    plt.figure(figsize=(45, 10))
    _ = tree.plot_tree(clf.best_estimator_, filled=True,
                       feature_names=X_train.feature_names, fontsize=12, precision=7, class_names=names)
    plt.show()


#########################################################
# SVM (LinearSVC) #######################################
#########################################################

# default parameters
def SVM_default(X_train, y_train, X_test, y_test, names):
    weight_1 = Counter(y_train)[0]/y_train.shape[0]*0.9
    weight_0 = 1 - weight_1
    # class_weight={0:weight_0, 1:weight_1}
    clf = LinearSVC(max_iter=10000, class_weight={0: weight_0, 1: weight_1})
    clf.fit(X_train, y_train)
    add_to_final_table(X_train, y_train, 'SVM_def', clf, names)
    add_to_final_table_test(X_test, y_test, 'SVM_def', clf, names)

# C=0.01

"""
# depreciated because hyperparameters fitting did not help to obtain better result 
def SVM_C(X_train, y_train, X_test, y_test, names):
    weight_1 = Counter(y_train)[0]/y_train.shape[0]*0.9
    weight_0 = 1 - weight_1
    clf = LinearSVC(C=0.01, max_iter=100000, class_weight={
                    0: weight_0, 1: weight_1})
    clf.fit(X_train, y_train)
    add_to_final_table(X_train, y_train, 'SVM_C', clf, names)
    add_to_final_table_test(X_test, y_test, 'SVM_C', clf, names)
"""
#########################################################
# RBF SVM ###############################################
#########################################################

# default parameters


def RBF_SVM(X_train, y_train, X_test, y_test, names):
    weight_1 = Counter(y_train)[0]/y_train.shape[0]*0.9
    weight_0 = 1 - weight_1
    clf = SVC(probability=True, class_weight={0: weight_0, 1: weight_1})
    clf.fit(X_train, y_train)
    add_to_final_table(X_train, y_train, 'RBF_SVM', clf, names)
    add_to_final_table_test(X_test, y_test, 'RBF_SVM', clf, names)

#########################################################
# LOGISTIC REGRESSION ###################################
#########################################################

# default parameters


def logistic_regression(X_train, y_train, X_test, y_test, names):
    weight_1 = Counter(y_train)[0]/y_train.shape[0]*0.9
    weight_0 = 1 - weight_1
    clf = LogisticRegressionCV(max_iter=10000, class_weight={
                               0: weight_0, 1: weight_1}).fit(X_train, y_train)
    add_to_final_table(X_train, y_train, 'LR', clf, names)
    add_to_final_table_test(X_test, y_test, 'LR', clf, names)
    feature_coef = pd.DataFrame(clf.coef_)
    feature_coef.columns = X_train.columns
    feature_coef.index = ['coef']
    feature_coef = feature_coef.T
    feature_coef_abs = abs(feature_coef['coef'])
    feature_coef_abs = feature_coef_abs.sort_values(ascending=False)
    feature_coef_abs_10 = feature_coef_abs[:10]
    most_important_features = feature_coef_abs_10.index

    engin_features = X_train.loc[:, X_train.columns.isin(['voting_age_awareness_w1',
                                                          'KNOWLEDGE_PARLIAMENTARY_THRESHOLD_w1',
                                                          'know_politicians_ratio',
                                                          'whether_dropped_before',
                                                          'lr_placement_correct',
                                                          'timeOfResponding',
                                                          'weekendResponse',
                                                          'whether_dropped_before',
                                                          'inconsistency',
                                                          'bad_quality',
                                                          'weekend',
                                                          'know_politicians_ratio',
                                                          'same_agree_resp',
                                                          'political_interest', 
                                                          'dont_know_percentage_mean',
                                                          'days_to_respond'])].columns

    def show_important(features, title):
        """filters the selected features with highest coefficients, adds title to the table"""
        df = feature_coef[feature_coef.index.isin(features)]
        df = df.style.set_table_attributes(
            "style='display:inline'").set_caption(title)
        return df

    df_m = show_important(most_important_features,
                          'The most important features and its coefficients obtained by logistic regression')
    df_b = show_important(engin_features, 'Engineered features coefficients')
    # displays one table on the left of another
    display_html(df_m._repr_html_()+df_b._repr_html_(), raw=True)


"""
ensemble models are omitted due to obtaining similar results as other models with longer waiting times

#########################################################
# XGBOOST ###############################################
#########################################################
# default parameters
def XGBoost_default(X_train, y_train, X_test, y_test, names):    
        
    model = xgboost.XGBClassifier(verbosity=0)
    y_pred = cross_val_predict(estimator=model, X=X_train, y=y_train, cv=5)
    XGBoost_to_final_table(y_pred, y_train, names)

# randomized search
def XGBoost_RS(X_train, y_train, X_test, y_test, names):
    model = xgboost.XGBClassifier(verbosity=0)
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
def RF_default(X_train, y_train, names):    
    
    clf = RandomForestClassifier()
    clf.fit(X_train, y_train)
    add_to_final_table(X_train, y_train, 'RF_default', clf, names)
    
# randomized search
def RF_random_search(X_train, y_train, names): 
    
    clf = RandomForestClassifier()
    random_grid = {'bootstrap': [True, False],
     'max_depth': [50, 100, 200],
     'min_samples_leaf': [5, 10, 20],
     'min_samples_split': [20, 40],
     'n_estimators': [500, 2000, 4000]}
    clf = RandomizedSearchCV(estimator = clf, param_distributions = random_grid, n_iter = 50, cv = 3, verbose=2, random_state=42, n_jobs = -1)
    # Fit the random search model
    clf.fit(X_train, y_train)
    print('Random_forest_best: ', clf.best_params_)
    add_to_final_table(X_train, y_train, 'RF_RS', clf, names)
"""


# In[ ]:


"""
def concat_df(df, political_data):
    personal_data = df.drop(['panelpat'], axis=1)
    df = pd.concat([personal_data, political_data], axis=1)
    return df
"""
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


# In[ ]:


def analysis_all_models(X_train, y_train, X_test, y_test, names):
    # XGBoost_RS(X_train, y_train)
    # XGBoost_default(X_train, y_train)
    # RF_default(X_train, y_train)
    # RF_random_search(X_train, y_train)
    # decision_tree_ccp_alpha(X_train, y_train, X_test, y_test)
    decision_tree(X_train, y_train, X_test, y_test, names)
    
    X_train, X_test = scale_train(X_train, X_test)
    SVM_default(X_train, y_train, X_test, y_test, names)
    # SVM_C(X_train, y_train, X_test, y_test, names)
    RBF_SVM(X_train, y_train, X_test, y_test, names)
    logistic_regression(X_train, y_train, X_test, y_test, names)


# In[ ]:


get_ipython().system('jupyter nbconvert --to script Analysis_functions.ipynb')


# In[ ]:




