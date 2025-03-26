# -*- coding: utf-8 -*-
"""
Created on Tue May 30 20:27:04 2023

@author: 24852
"""
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from LAMDA_SSL.Algorithm.Classification.LapSVM import LapSVM
import pickle
from LAMDA_SSL.Algorithm.Classification.SemiBoost import SemiBoost
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier

from sklearn.svm import SVC
from LAMDA_SSL.Split.ViewSplit import ViewSplit
from LAMDA_SSL.Algorithm.Classification.Tri_Training import Tri_Training
from LAMDA_SSL.Algorithm.Classification.Co_Training import Co_Training
from LAMDA_SSL.Algorithm.Classification.Assemble import Assemble
from LAMDA_SSL.Algorithm.Classification.TSVM import TSVM
from LAMDA_SSL.Algorithm.Classification.SSGMM import SSGMM


def semiboost(labeled_X, labeled_y, unlabeled_X):
    base_estimator = SVC(C=100, kernel='sigmoid', probability=True, gamma='auto', random_state=22)

    model = SemiBoost(T=1, similarity_kernel='rbf', base_estimator=base_estimator)
    model.fit(X=labeled_X, y=labeled_y, unlabeled_X=unlabeled_X)
    pred_y = model.predict(unlabeled_X)
    prob_y = model.predict_proba(unlabeled_X)

    return pred_y, prob_y,model


def tri_training(labeled_X, labeled_y, unlabeled_X):

    base_estimator = SVC(C=100, kernel='rbf', probability=True, random_state=20)
    base_estimator_2 = SVC(C=50, kernel='sigmoid', probability=True, random_state=40)
    base_estimator_3 = SVC(C=25, kernel='sigmoid', probability=True, random_state=25)
    model = Tri_Training(base_estimator=base_estimator, base_estimator_2=base_estimator_2,
                         base_estimator_3=base_estimator_3)
    model.fit(X=labeled_X, y=labeled_y, unlabeled_X=unlabeled_X)

    # #%%% 预测
    pred_y = model.predict(unlabeled_X)
    prob_y = model.predict_proba(unlabeled_X)


    return pred_y,prob_y,model


def co_training(labeled_X, labeled_y, unlabeled_X):

    split_labeled_X = ViewSplit(labeled_X, shuffle=False)
    split_unlabeled_X = ViewSplit(unlabeled_X, shuffle=False)
    split_test_X = ViewSplit(unlabeled_X, shuffle=False)

    base_estimator = SVC(C=50, kernel='rbf', probability=True, random_state=11)
    base_estimator_2 = SVC(C=50, kernel='sigmoid', probability=True, random_state=28)
    model = Co_Training(base_estimator=base_estimator, base_estimator_2=base_estimator_2,
                        s=( len(labeled_X) + len(unlabeled_X)) // 10,random_state=33)

    model.fit(X=split_labeled_X, y=labeled_y, unlabeled_X=split_unlabeled_X)


    pred_y = model.predict(split_test_X)
    prob_y = model.predict_proba(split_test_X)

    return pred_y,prob_y,model


def lapsvm(labeled_X, labeled_y, unlabeled_X):
    model = LapSVM(gamma_A=0.001, gamma_I=0.5)
    model.fit(X=labeled_X, y=labeled_y, unlabeled_X=unlabeled_X)
    train_performance = model.evaluate(labeled_X, labeled_y)
    pred_y = model.predict(unlabeled_X)
    prob_y = model.predict_proba(unlabeled_X)
    return pred_y,prob_y,model


def assemble(labeled_X, labeled_y, unlabeled_X):

    base_estimator = SVC(probability=True, C=1, gamma='auto', kernel='rbf', random_state=22)
    model = Assemble(base_estimator=base_estimator)
    model.fit(X=labeled_X, y=labeled_y, unlabeled_X=unlabeled_X)
    pred_y = model.predict(unlabeled_X)
    prob_y = model.predict_proba(unlabeled_X)
    return pred_y,prob_y,model


def tsvm(labeled_X, labeled_y, unlabeled_X):
    model = TSVM(Cl=50, Cu=0.001, degree=2, gamma='auto', kernel='rbf')
    model.fit(X=labeled_X, y=labeled_y, unlabeled_X=unlabeled_X)

    # #%%% 预测
    pred_y = model.predict(unlabeled_X)
    prob_y = model.predict_proba(unlabeled_X)


    return pred_y,prob_y,model


def ssgmm(labeled_X, labeled_y, unlabeled_X):
    model = SSGMM(tolerance=.001)

    # random_search = BayesSearchCV(model, param_distributions=param_dict,scoring='accuracy',n_jobs=-1,cv=10)
    model.fit(X=labeled_X, y=labeled_y, unlabeled_X=unlabeled_X)
    # #%%% 预测
    pred_y = model.predict(unlabeled_X)
    prob_y = model.predict_proba(unlabeled_X)

    return pred_y,prob_y,model
def svc(labeled_X, labeled_y, unlabeled_X):
    model = SVC(C=100,gamma=0.001,kernel='sigmoid',random_state=20,probability=True)
    model.fit(X=labeled_X, y=labeled_y)
    # #%%% 预测
    pred_y = model.predict(unlabeled_X)
    prob_y = model.predict_proba(unlabeled_X)

    return pred_y,prob_y,model
def rf(labeled_X, labeled_y, unlabeled_X,seed=22):
    rf_params = {'max_depth': 10, 'max_features': 0.999,
                 'min_samples_leaf': 25, 'min_samples_split': 25,
                 'n_estimators': 152}
    model = RandomForestClassifier(**rf_params, random_state=seed)
    model.fit(X=labeled_X, y=labeled_y)
    # #%%% 预测
    pred_y = model.predict(unlabeled_X)
    prob_y = model.predict_proba(unlabeled_X)

    return pred_y,prob_y,model
def LR(labeled_X, labeled_y, unlabeled_X,seed=22):
    model = LogisticRegression(random_state=seed)
    model.fit(X=labeled_X, y=labeled_y)
    # #%%% 预测
    pred_y = model.predict(unlabeled_X)
    prob_y = model.predict_proba(unlabeled_X)

    return pred_y,prob_y,model
def Xgboost(labeled_X, labeled_y, unlabeled_X,seed=22):
    xgb_params = {'colsample_bytree': 0.01, 'learning_rate': 0.09349801051326335,
                  'max_depth': 5, 'min_child_weight': 6,
                  'n_estimators': 65, 'subsample': 0.8715969639120398}
    model = xgb.XGBClassifier(**xgb_params,random_state=seed)
    model.fit(X=labeled_X, y=labeled_y)
    # #%%% 预测
    pred_y = model.predict(unlabeled_X)
    prob_y = model.predict_proba(unlabeled_X)

    return pred_y,prob_y,model
def GDBT(labeled_X, labeled_y, unlabeled_X,seed=22):
    GDBT_params = {'learning_rate': 0.001, 'max_depth': 3,
                   'max_features': 0.1, 'min_samples_leaf': 1, 'min_samples_split': 14,
                   'n_estimators': 136}
    model = GradientBoostingClassifier(**GDBT_params,random_state=seed)
    model.fit(X=labeled_X, y=labeled_y)
    # #%%% 预测
    pred_y = model.predict(unlabeled_X)
    prob_y = model.predict_proba(unlabeled_X)

    return pred_y,prob_y,model