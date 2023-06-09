import numpy as np
from scipy.stats.mstats import gmean
from sklearn.metrics import mean_squared_error, make_scorer, r2_score
from sklearn.model_selection import cross_val_score, train_test_split, StratifiedKFold
from sklearn.linear_model import LogisticRegression 
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import pandas as pd
from sklearn.svm import SVC
from sklearn.metrics import balanced_accuracy_score
from sklearn.inspection import permutation_importance
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

n_jobs = 6

def my_closure(mat):
    mat = np.atleast_2d(mat)
    if np.any(mat < 0):
        raise ValueError("Cannot have negative proportions")
    if mat.ndim > 2:
        raise ValueError("Input matrix can only have two dimensions or less")
    mat = mat / mat.sum(axis=1, keepdims=True)
    return mat.squeeze()

def my_multiplicative_replacement(mat,delta=None):
    mat = my_closure(mat)
    z_mat = (mat == 0)

    num_feats = mat.shape[-1]
    tot = z_mat.sum(axis=-1, keepdims=True)

    if delta is None:
        delta = (1. / num_feats)**2

    zcnts = 1 - tot * delta
    mat = np.where(z_mat, delta, zcnts * mat)
    return mat.squeeze()

def my_clr(mat):
    #mat = my_closure(mat)
    #lmat = np.log(mat)
    #gm = lmat.mean(axis=-1, keepdims=True)
    #return (lmat - gm).squeeze()
    return np.log(mat) - np.log(gmean(mat))

def remove_rows_where_column_nan(df,column_1,column_2=None):
    if column_2 == None:
        return df[df[column_1].notna()]
    elif column_2 != None:
        df_tmp = df[df[column_1].notna()]
        return df_tmp[df_tmp[column_2].notna()]

def remove_rows_where_column_specified(df,column_1,specified_1,column_2=None,specified_2=None,column_3=None,specified_3=None):
    if column_2 == None and specified_2 == None:
        return df.drop(df.loc[df[column_1]==specified_1].index)
    elif column_2 != None and specified_2 != None and (column_3 == None and specified_3 == None):
        df_tmp = df.drop(df.loc[df[column_1]==specified_1].index)
        return df_tmp.drop(df_tmp.loc[df_tmp[column_2]==specified_2].index)
    elif column_3 != None and specified_3 != None:
        df_tmp = df.drop(df.loc[df[column_1]==specified_1].index)
        df_tmp_tmp = df_tmp.drop(df_tmp.loc[df_tmp[column_2]==specified_2].index)
        return df_tmp_tmp.drop(df_tmp_tmp.loc[df_tmp_tmp[column_3]==specified_3].index)

def keep_rows_where_column_specified(df,column,specified):
    return df.loc[df[column] == specified]

def remove_rows_where_column_larger_than(df,column,value):
    return df.drop(df.loc[df[column]>value].index)

def remove_rows_where_column_lower_than(df,column,value):
    return df.drop(df.loc[df[column]<value].index)

def rename_columns(X, list_bool = False):
    if list_bool == False:
        new_columns = [f"feature_{i}" for i in range(len(X.columns))]
        X.columns = new_columns
    elif list_bool == True:
        X = [f"feature_{i}" for i in range(len(X))]
    return X

def SVM_features(X, y, C, gamma):
    #Final model
    model = SVC(kernel = 'rbf', C=C, gamma=gamma)
    model.fit(X,y)
    yhat = model.predict(X)
    # Feature importance
    coef = permutation_importance(model, X, y, n_repeats=5, random_state=1, n_jobs=n_jobs)
    feature_importance = pd.DataFrame({'Feature': X.columns, 'Importance': coef.importances_mean})
    return feature_importance, yhat

def logistic_features(X, y, C, penalty):
    #Final model
    model = LogisticRegression(solver='saga', C=C, penalty=penalty, max_iter=200,  multi_class='multinomial', random_state=1, n_jobs = n_jobs)
    model.fit(X,y)
    yhat = model.predict(X)
    # Feature importance
    #logreg_coefs = model.coef_
    coef = np.sum(abs(model.coef_),axis=0)
    feature_importance = pd.DataFrame({'Feature': X.columns, 'Importance': coef})
    return feature_importance, yhat

def rf_features(X, y, max_features, n_estimators):
    #Final model
    model = RandomForestClassifier(max_features=max_features, n_estimators=n_estimators, random_state=1, criterion='gini', n_jobs = n_jobs)
    model.fit(X,y)
    # Feature importance
    yhat = model.predict(X)
    coef = model.feature_importances_
    feature_importance = pd.DataFrame({'Feature': X.columns, 'Importance': coef})
    return feature_importance, yhat

def XGB_features(X, y, n_estimators, max_depth, learning_rate):
    X_cols = X.columns
    #Final model
    X = rename_columns(X)
    le = LabelEncoder()
    y_XGB = pd.Series(le.fit_transform(y))
    model = XGBClassifier(n_estimators, max_depth, learning_rate, colsample_bytree=0.02, reg_lambda=1, objective='multi:softmax')
    model.fit(X,y_XGB)
    yhat = le.inverse_transform(model.predict(X))
    # Feature importance
    coef = model.feature_importances_
    feature_importance = pd.DataFrame({'Feature': X_cols, 'Importance': coef})
    return feature_importance, yhat

# Plot functions

SMALL_SIZE = 6
MEDIUM_SIZE = 6
MEDIUMBIG_SIZE = 10
BIGGER_SIZE = 16

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=MEDIUMBIG_SIZE)    # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

def plot_distribution(y, x_ticks_order, title, width=6, height=6):
    frequencies = [y.count(element) for element in x_ticks_order]
    plt.figure(figsize=(width,height))
    plt.bar(x_ticks_order, frequencies)
    plt.ylabel('Frequency')
    plt.title(title)
    plt.xticks(rotation=30, ha='right') 
    plt.show()

def plot_top_features(feature_importances, feature_names, top_k):
    # Sort feature importances in descending order and select the top features
    indices = np.argsort(feature_importances)[::-1][:top_k]
    top_features = feature_names[indices]
    top_importances = feature_importances[indices]
    # Plotting the top features as bars
    plt.figure(figsize=(8, 5))
    plt.barh(range(len(top_features)), top_importances, align='center')
    plt.yticks(range(len(top_features)), top_features)
    plt.xlabel('Feature Importance')
    plt.ylabel('Feature')
    plt.title('Top {} Feature Importances'.format(top_k))
    plt.tight_layout()
    plt.show()

def plot_top_features_subplot(feature_importances_dict, feature_names, top_k, main_title):
    # Determine the number of methods and features
    num_methods = len(feature_importances_dict)
    num_features = len(next(iter(feature_importances_dict.values())))

    # Create subplots
    fig, axs = plt.subplots(num_methods, 1, figsize=(6, 2.2 * num_methods))

    # Get the top features for each method
    top_features_per_method = {}
    for method, importances in feature_importances_dict.items():
        indices = np.argsort(importances)[::-1][:top_k]
        top_features_per_method[method] = set(np.array(feature_names)[indices.tolist()])  # Convert indices to list

    # Count the occurrences of each feature in the top lists
    feature_counts = {}
    for features in top_features_per_method.values():
        for feature in features:
            feature_counts[feature] = feature_counts.get(feature, 0) + 1

    # Plotting the top features for each method with different colors for overlapping features
    for i, (method, importances) in enumerate(feature_importances_dict.items()):
        # Sort feature importances in descending order and select the top features
        indices = np.argsort(importances)[::-1][:top_k]
        top_features = np.array(feature_names)[indices]
        top_importances = importances[indices]

        # Assign colors to features based on their occurrence in top lists
        colors = ['tab:blue' if feature_counts[feature] == 1 else 'tab:red' if feature_counts[feature] == 2 else 'tab:green' for feature in top_features]
        fig.suptitle(main_title)
        # Plot the top features as bars with assigned colors
        axs[i].barh(range(len(top_features)), top_importances, color=colors, align='center')
        axs[i].set_yticks(range(len(top_features)))
        axs[i].set_yticklabels(top_features)
        axs[i].set_xlabel('Feature Importance')
        axs[i].set_ylabel('Feature')
        axs[i].set_title(f'Top {top_k} Feature Importances - {method}')

    plt.tight_layout()
    plt.show()
