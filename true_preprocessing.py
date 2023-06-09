import pandas as pd
from functions import *

X_data = pd.read_csv('X_data.csv', low_memory=False)
X_data = X_data.rename(columns = {'Unnamed: 0':'id'}) #Changing name of id-column
X_data = X_data.loc[:,(X_data != 0).any(axis=0)] #Removing all all-zero columns

meta_data = pd.read_csv('meta_data.csv', low_memory=False)
meta_data = meta_data.rename(columns = {'Unnamed: 0':'id'}) #Changing name of id-column
X_data_noid = X_data.iloc[:,1:]

# Keeps columns with at least 25% non-zero values
list = []
for column_name in X_data_noid.columns:
    column = X_data_noid[column_name]
    share = 1-(column == 0).sum()/len(X_data_noid)
    if share >= 0.53:
        list.append(column_name)
X_1 = X_data_noid[list]

# Keeps columns with median of non-zero values bigger than x
list = []
for column_name in X_1.columns:
    if column_name != 'id':
        non_zeros = X_1.loc[X_1[column_name] != 0, column_name]
        if non_zeros.median() >= 4:
            list.append(column_name)
    else:
        list.append(column_name)
X_2 = X_1[list]
# Add constant to avoid log of zero
X_3 = X_2+1
# Project to simplex
X_4 = X_3 / np.sum(X_3, axis=1)[:, np.newaxis]
# Attaches 'id' and performs clr
X = pd.concat((X_data.loc[:,'id'],my_clr(X_4.astype(float))),axis=1)