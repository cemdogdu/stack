
from copy import deepcopy
from itertools import combinations

import numpy as np
import pandas as pd

from tqdm import tqdm
from scipy.stats import norm, chi2
from scipy.stats import t as t_dist
from sklearn.datasets import load_wine
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, KFold

# Libs implementations
from mlxtend.evaluate import mcnemar
from mlxtend.evaluate import mcnemar_table
from mlxtend.evaluate import paired_ttest_5x2cv
from mlxtend.evaluate import proportion_difference
from mlxtend.evaluate import paired_ttest_kfold_cv
from mlxtend.evaluate import paired_ttest_resampled

import pandas as pd
import numpy as np
import glob
import os
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn import ensemble
from sklearn import svm
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler


# Define paired t-test function //make sure it is two-sided
def paired_t_test(p):
    p_hat = np.mean(p)
    n = len(p)
    den = np.sqrt(sum([(diff - p_hat) ** 2 for diff in p]) / (n - 1))
    t = (p_hat * (n ** (1 / 2))) / den

    p_value = t_dist.sf(abs(t), n - 1) * 2

    return t, p_value


# Data
file_list = glob.glob(r'C:\Users\cemdogdu\PycharmProjects\wekaapi\datasets\*.csv*')  ##put the all csv data files in a folder
path = r'C:\Users\cemdogdu\PycharmProjects\wekaapi\datasets\*.csv*'

print('*' * 50)
print('>>> file_list:', file_list)
print('*' * 50)

for file in glob.glob(path):
    data = pd.read_csv(file)
    # print(data)
    column_list = list(data.columns)
    X = data[column_list[1:-1]]  # Features
    y = data['emotion']  # Labels
    # print('>>>features:', column_list[1:-1])
    # print('*'*50)
    # print ('---------data--------',X)
    # print('*'*50)
    # print(y)
    # print('*'*50)

# Instantiating the classification algorithms
    mlp_clf = MLPClassifier(solver='adam', alpha=1e-5, hidden_layer_sizes=(256, 128, 64, 32), random_state=42)
    rf_clf  = RandomForestClassifier(random_state=42, n_estimators=100)
    knn_clf = KNeighborsClassifier(n_neighbors=1)
    dt_clf  = tree.DecisionTreeClassifier(max_depth=5, random_state=42)
    gb_clf  = ensemble.GradientBoostingClassifier(random_state=42, n_estimators=100)
    svm_clf = svm.SVC(kernel='linear', C=1, random_state=42)

    # For holdout cases
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

    #create clf lists
    clf_list = [rf_clf, knn_clf, dt_clf, gb_clf, svm_clf, mlp_clf]
    clf_list_name = ['rf', 'knn', 'dt', 'gb', 'svm','mlp']

    # ------------------ paired resampled t TEST-----------with defined function on accuracies---   #

    print('*' * 50)
    print('------------------ t TEST----------------------')
    print('*' * 50)


    for ii in range(len(clf_list)):
        for jj in range(len(clf_list)):

            print("Paired t-test Resampled")
            t, p = paired_ttest_resampled(estimator1=clf_list[ii], estimator2=clf_list[jj], X=X, y=y, random_seed=42, num_rounds=30, test_size=0.2)
            print(ii,jj)
            print(clf_list[ii], clf_list[jj])
            print(f"t statistic: {t}, p-value: {p}\n")







    '''
    for kk in range(len(acc_list)):
        for pp in range(len(acc_list)):
            p_.append(kk - pp)

            print("Paired t-test Resampled")
            print(file)
            print(acc_list[kk], acc_list[pp])
            t, p = paired_t_test(p_)
            print(f"t statistic: {t}, p-value: {p}\n")

     '''





    """
       # for ii in range(len(acc_list)):
        #     for jj in range(ii + 1, len(acc_list)):
        #         print(acc_list[ii], acc_list[jj])

    if ii == 0 and jj != 0:
        #normalization
        scaler = StandardScaler()
        scaler.fit(X_train)
        X_train_tr = scaler.transform(X_train)
        #train
        clf1.fit(X_train_tr, y_train)
        clf2.fit(X_train, y_train)
    if ii != 0 and jj == 0:
        scaler = StandardScaler()
        scaler.fit(X_train)
        X_train_tr = scaler.transform(X_train)
        clf1.fit(X_train, y_train)
        clf2.fit(X_train_tr, y_train)
    else:
        clf1.fit(X_train, y_train)
        clf2.fit(X_train, y_train)
    # # Accuracies

    # Fit the models
    if ii == 0 and jj != 0:
        X_test_tr = scaler.transform(X_test)
        acc_clf1 = accuracy_score(y_test, clf1.predict(X_test_tr))
        acc_clf2 = accuracy_score(y_test, clf2.predict(X_test))
    if ii != 0 and jj == 0:
        X_test_tr = scaler.transform(X_test)

        acc_clf1 = accuracy_score(y_test, clf1.predict(X_test))
        acc_clf2 = accuracy_score(y_test, clf2.predict(X_test_tr))
    else:
        acc_clf1 = accuracy_score(y_test, clf1.predict(X_test))
        acc_clf2 = accuracy_score(y_test, clf2.predict(X_test))

    # # Create list of accuracies
    # # Append t-test function for each pair of accuracy scores
    """




    """
 ###COMPARISON TESTS###

     # Two Proportions Test  ###problem: assumes independent samples but in our case it is not !
     # first take combinations *2


     print('*' * 50)
     print('------------------ Z TEST----------------------')
     print('*' * 50)

     accuracy_list = [acc_dt, acc_rf, acc_gb, acc_svm, acc_mlp]
     accuracy_list_name = ['acc_dt', 'acc_rf', 'acc_gb', 'acc_svm', 'acc_mlp']

     p1 = np.zeros((len(accuracy_list), len(accuracy_list)), dtype=np.float64)
     z1 = np.zeros((len(accuracy_list), len(accuracy_list)), dtype=np.float64)

     for ii in range(len(accuracy_list)):
         for jj in range(ii + 1, len(accuracy_list)):
             print('ii: ', ii, '-jj: ', jj)
             print('dataset:',file)
             print('->Comparison of: ' + accuracy_list_name[ii] + ' -- ' + accuracy_list_name[jj])

             z1[ii, jj], p1[ii, jj] = proportion_difference(accuracy_list[ii], accuracy_list[jj], n_1=len(y_test))

             print(f"z statistic: {z1[ii, jj]}, p-value: {p1[ii, jj]}\n")
             print(accuracy_list_name[ii],":",accuracy_list[ii])
             print(accuracy_list_name[jj],":",accuracy_list[jj])
             print('*' * 50)


     #------------------ paired resampled t TEST-----------Mlxtend library-----------with loop on classifiers which is not so timely efficient---
     print('*' * 50)
     print('------------------paired resampled t TEST-----------Mlxtend library-------with loop------')
     print('*' * 50)

     clf_list = [dt_clf, rf_clf, gb_clf, svm_clf, clf_mlp]
     clf_list_name = ['dt', 'rf', 'gb', 'svm', 'mlp']

     p2 = np.zeros((len(clf_list), len(clf_list)), dtype=np.float64)
     t1 = np.zeros((len(clf_list), len(clf_list)), dtype=np.float64)


     for kk in range(len(clf_list)):
         for pp in range(kk + 1, len(clf_list)):
             print('kk: ', kk, '-pp: ', pp)
             print('dataset:', file)
             print('->Comparison of: ' + clf_list_name[kk] + ' -- ' + clf_list_name[pp])
             t1[kk,pp], p2[kk,pp] = paired_ttest_resampled(estimator1=clf_list[kk], estimator2=clf_list[pp], X=X, y=y,
                                                              random_seed=42, num_rounds=10,
                                                              test_size=0.2)

             print(f"t statistic: {t1[kk,pp]}, p-value: {p2[kk,pp]}\n")

             #t1[kk, pp], p2[kk, ] = proportion_difference(accuracy_list[ii], accuracy_list[jj], n_1=len(y_test))
             #print(clf_list[kk])
             #print(clf_list[pp])
     """
