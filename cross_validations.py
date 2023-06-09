from functions import *

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression 
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import balanced_accuracy_score
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC

n_jobs = 2
scorer = make_scorer(balanced_accuracy_score)

def dummy_CV(X, y):
    ref_scores = []
    for i in range(7):
        kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)
        dummy = DummyClassifier(strategy="prior")
        ref_scores.append(np.mean(cross_val_score(dummy, X, y, scoring=scorer, cv=kf)))
    return ref_scores

def SVM_CV(X, y):
    # Define outer cross-validation splits
    cv_outer = StratifiedKFold(n_splits=7, shuffle=True, random_state=1)

    # Create lists to store results
    results_params = []; results_scores = []

    # Perform nested cross-validation
    for train_ix, test_ix in cv_outer.split(X,y):
        # Split data
        X_train, X_test = X.iloc[train_ix, :], X.iloc[test_ix, :]
        y_train, y_test = y.iloc[train_ix], y.iloc[test_ix]
        
        # Configure the inner cross-validation procedure
        cv_inner = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)
        
        # Define the model and parameters
        model = SVC(random_state=1)
        parameters = {"C": [1, 10, 100], "gamma": [.001, .01, .1]}

        # Define and do search
        search = GridSearchCV(model, parameters, scoring=scorer, cv=cv_inner, refit=True, n_jobs = n_jobs)
        result = search.fit(X_train, y_train)
        
        # Get the best performing model fit on the whole training set and evaluate it on the holdout set
        best_model = result.best_estimator_
        yhat = best_model.predict(X_test)
        
        # Evaluate the model
        acc = balanced_accuracy_score(y_test, yhat)
        
        # Report progress
        print('>acc=%.3f, est=%.3f, cfg=%s' % (acc, result.best_score_, result.best_params_))
        #outer_results.append([acc, result.best_score_, result.best_params_])
        results_scores.append(acc)
        results_params.append(result.best_params_)
    return results_scores, results_params

    # Summarize the estimated performance of the model
    #print('Accuracy: %.3f (%.3f)' % (np.mean(outer_results), np.std(outer_results)))

def logistic_CV(X, y):
    # Define outer cross-validation splits
    cv_outer = StratifiedKFold(n_splits=7, shuffle=True, random_state=1)

    # Create lists to store results
    results_params = []; results_scores = []

    # Perform nested cross-validation
    for train_ix, test_ix in cv_outer.split(X,y):
        # Split data
        X_train, X_test = X.iloc[train_ix, :], X.iloc[test_ix, :]
        y_train, y_test = y.iloc[train_ix], y.iloc[test_ix]
        
        # Configure the inner cross-validation procedure
        cv_inner = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)
        
        # Define the model and parameters
        model = LogisticRegression(solver='saga', max_iter=200,  multi_class='multinomial', random_state=1, n_jobs = n_jobs)
        parameters = {'penalty':['l1','l2'], 'C':list(np.logspace(-4,2,7,10))}

        # Define and do search
        search = GridSearchCV(model, parameters, scoring=scorer, cv=cv_inner, refit=True, n_jobs = n_jobs)
        result = search.fit(X_train, y_train)
        
        # Get the best performing model fit on the whole training set and evaluate it on the holdout set
        best_model = result.best_estimator_
        yhat = best_model.predict(X_test)
        
        # Evaluate the model
        acc = balanced_accuracy_score(y_test, yhat)
        
        # Report progress
        print('>acc=%.3f, est=%.3f, cfg=%s' % (acc, result.best_score_, result.best_params_))
        results_scores.append(acc)
        results_params.append(result.best_params_)
    return results_scores, results_params

def rf_CV(X, y):
    # Define outer cross-validation splits
    cv_outer = StratifiedKFold(n_splits=7, shuffle=True, random_state=1)

    # Create lists to store results
    results_params = []; results_scores = []

    # Perform nested cross-validation
    for train_ix, test_ix in cv_outer.split(X,y):
        # Split data
        X_train, X_test = X.iloc[train_ix, :], X.iloc[test_ix, :]
        y_train, y_test = y.iloc[train_ix], y.iloc[test_ix]
        
        # Configure the inner cross-validation procedure
        cv_inner = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)
        
        # Define the model and parameters
        model = RandomForestClassifier(random_state=1, criterion='gini', n_jobs = n_jobs)
        parameters = {'n_estimators':np.linspace(50,1000,3,True,dtype=int), 'max_features':np.linspace(0.05,0.9,3,True)}

        # Define and do search
        search = GridSearchCV(model, parameters, scoring=scorer, cv=cv_inner, refit=True, n_jobs = n_jobs)
        result = search.fit(X_train, y_train)
        
        # Get the best performing model fit on the whole training set and evaluate it on the holdout set
        best_model = result.best_estimator_
        yhat = best_model.predict(X_test)
        
        # Evaluate the model
        acc = balanced_accuracy_score(y_test, yhat)
        
        # Report progress
        print('>acc=%.3f, est=%.3f, cfg=%s' % (acc, result.best_score_, result.best_params_))
        results_scores.append(acc)
        results_params.append(result.best_params_)
    return results_scores, results_params

def XGB_CV(X, y):
    #Transform classes to numbers and feature names to feature_0, feature_1 and so on
    le = LabelEncoder()
    y_XGB = pd.Series(le.fit_transform(y))
    X = rename_columns(X)

    # Define outer cross-validation splits
    cv_outer = StratifiedKFold(n_splits=7, shuffle=True, random_state=1)

    # Create lists to store results
    results_params = []; results_scores = []

    # Perform nested cross-validation
    for train_ix, test_ix in cv_outer.split(X,y):
        # Split data
        X_train, X_test = X.iloc[train_ix, :], X.iloc[test_ix, :]
        y_train, y_test = y_XGB.iloc[train_ix], y_XGB.iloc[test_ix]
        
        # Configure the inner cross-validation procedure
        cv_inner = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)
        
        # Define the model and parameters
        model = XGBClassifier(colsample_bytree=0.02, reg_lambda=1, objective='multi:softmax')
        parameters = {'max_depth': [2, 4, 8],
              'learning_rate': [0.03, 0.1, 0.3],
              #'subsample': np.arange(0.5, 1.0, 0.1),
              #'colsample_bytree': np.arange(0.4, 1.0, 0.2),
              #'colsample_bylevel': np.arange(0.4, 1.0, 0.2),
              'n_estimators': [100, 500, 1000]}

        # Define and do search
        search = GridSearchCV(model, parameters, scoring=scorer, cv=cv_inner, refit=True, n_jobs = n_jobs)
        result = search.fit(X_train, y_train)
        
        # Get the best performing model fit on the whole training set and evaluate it on the holdout set
        best_model = result.best_estimator_
        yhat = best_model.predict(X_test)
        
        # Evaluate the model
        acc = balanced_accuracy_score(y_test, yhat)
        
        # Report progress
        print('>acc=%.3f, est=%.3f, cfg=%s' % (acc, result.best_score_, result.best_params_))
        results_scores.append(acc)
        results_params.append(result.best_params_)
    return results_scores, results_params
