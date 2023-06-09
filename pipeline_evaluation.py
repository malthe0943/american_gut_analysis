import pandas as pd
from functions import *

def get_data():
    X_data = pd.read_csv('X_data.csv', low_memory=False)
    X_data = X_data.rename(columns = {'Unnamed: 0':'id'}) #Changing name of id-column
    X_data = X_data.loc[:,(X_data != 0).any(axis=0)] #Removing all all-zero columns

    meta_data = pd.read_csv('meta_data.csv', low_memory=False)
    meta_data = meta_data.rename(columns = {'Unnamed: 0':'id'}) #Changing name of id-column
    X_data_noid = X_data.iloc[:,1:]
    return X_data_noid, meta_data

def preprocess(X, meta_data, percent_nonzero = 0.25, median_threshold = 5):

    #Keeps columns with at least 25% non-zero values
    list = []
    for column_name in X.columns:
        column = X[column_name]
        share = 1-(column == 0).sum()/len(X)
        if share >= percent_nonzero:
            list.append(column_name)
    X_1 = X[list]

    #Keeps columns with median of non-zero values bigger than x
    list = []
    for column_name in X_1.columns:
        if column_name != 'id':
            non_zeros = X_1.loc[X_1[column_name] != 0, column_name]
            if non_zeros.median() >= median_threshold:
                list.append(column_name)
        else:
            list.append(column_name)
    X_2 = X_1[list]
    X_3 = X_2+1
    X_4 = X_3 / np.sum(X_3, axis=1)[:, np.newaxis]
    #X = pd.concat((X.loc[:,'id'],my_clr(X_4.astype(float))),axis=1)
    X = my_clr(X_4.astype(float))
    return X, meta_data