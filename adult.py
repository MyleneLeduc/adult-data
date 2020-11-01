# -*- coding: utf-8 -*-
"""
Created on Thu Oct 15 17:53:41 2020

@author: Mylène
"""


import pandas as pd
import os
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder
from evaluate import evaluate_model
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from data import read_data_file
from reduce_skewness import ReduceSkewness
from Encoder import One_Hot_Encoder

# Read file

train_df = read_data_file('adult.data')
test_df = read_data_file('adult.test')

# Drop the fnlwgt column which is which is useless for the later analysis

train_df = train_df.drop('fnlwgt', axis=1)
test_df = test_df.drop('fnlwgt', axis=1)

# Get list of categorical variable

object_col = train_df.select_dtypes(include=object).columns.tolist()
for col in object_col:
    print(train_df[col].value_counts(dropna=False)/train_df.shape[0],'\n')


# Convert '?' to NANs

point_inter = train_df.loc[26,'workclass']
for col in object_col:
    train_df.loc[train_df[col]==point_inter, col] = np.nan
    test_df.loc[test_df[col]==point_inter, col] = np.nan


# Perform an missing assessment in each column of the dataset

col_missing_pct = train_df.isna().sum()/train_df.shape[0]
col_missing_pct.sort_values(ascending=False)


# Remove data entries with missing values

train_df = train_df.dropna(axis=0,how='any')
test_df = test_df.dropna(axis=0,how='any')

# Show the result of the split

print("After removing the missing values:")
print("Training set has {} values.".format(train_df.shape[0]))
print("Training set has {} values.".format(test_df.shape[0]))

### Calculate skew and sort ###

mycolumns = ['capital-gain','capital-loss','age','hours-per-week','education-num']
skew_feats = train_df[mycolumns].skew().sort_values(ascending=False)
skewness = pd.DataFrame({'skew':skew_feats})
#print("Skewness: {:.4f}".format(skewness))

# Reduction of skewness by taking logarithme of variable
train_df = ReduceSkewness(train_df,['capital-gain','capital-loss'])
test_df = ReduceSkewness(test_df,['capital-gain','capital-loss'])

# Initialize a scaler then apply it to the features

scaler = MinMaxScaler()     # default = (0,1)

features_log_min_max_transform = pd.DataFrame(data = train_df)
features_log_min_max_transform[mycolumns] = scaler.fit_transform(train_df[mycolumns])

# Transform the test dataset
features_log_min_max_transform_test = pd.DataFrame(data = test_df)
features_log_min_max_transform_test[mycolumns] = scaler.fit_transform(test_df[mycolumns])

### Data preprocessing : One-hot encoding ###

object_cols = [
        'workclass',
        'education',
        'marital-status',
        'occupation',
        'relationship',
        'race',
        'sex',
        'native-country',
    ]

# Apply one-hot encoder to each column with categorical data


OH_encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)
categorical_feats_train = train_df[object_cols]
index_train = categorical_feats_train.index
categorical_feats_test = test_df[object_cols]
index_test = categorical_feats_test.index

OH_cols_train = OH_encoder.fit_transform(categorical_feats_train)
OH_cols_test = OH_encoder.transform(categorical_feats_test)


# Add one-hot encoded columns to numerical features
name_columns = OH_encoder.get_feature_names(object_cols)

OH_cols_train = pd.DataFrame(OH_cols_train, columns = name_columns)
OH_cols_test = pd.DataFrame(OH_cols_test, columns = name_columns)

# One-hot encoding index
OH_cols_train.index = categorical_feats_train.index
OH_cols_test.index = categorical_feats_test.index

### Concatenate numerical features and encoded categorical features together

# Concatenate on train dataset

X_train = pd.merge(features_log_min_max_transform[mycolumns],OH_cols_train, left_index=True, right_index=True)

income_raw_train = train_df['salary']
Y_train = income_raw_train.apply(lambda x:1 if x==' >50K' else 0)

# Concatenate on test dataset

X_test = pd.merge(features_log_min_max_transform_test[mycolumns],OH_cols_test, left_index=True,
                   right_index=True)
income_raw_test = test_df['salary']
Y_test = income_raw_test.apply(lambda x:1 if x==' >50K.' else 0)

### Evaluate model performance ###

# Naive predictor

pred_naive = np.ones(len(Y_test))

evaluate_naive = evaluate_model(Y_test,pred_naive)
erreur_naive = sum(abs(pred_naive-Y_test))/len(Y_test)
print("Naive Predictor : Erreur : {:.4f}, Accuracy score : {:.4f}, F-score : {:.4f}".format(erreur_naive,evaluate_naive[0],evaluate_naive[1]))

# Random forest method

forest_model = RandomForestRegressor(random_state=1)

forest_model.fit(X_train, Y_train)
y_pred_forest = forest_model.predict(X_test)
y_test = Y_test.to_numpy()

erreur_forest = sum(abs(y_pred_forest-y_test))/len(y_test)  # erreur de 20 %
evaluate_tree = evaluate_model(Y_test,y_pred_forest)
print("Forest Predictor : Erreur : {:.4f}, Accuracy score : {:.4f}, F-score : {:.4f}".format(erreur_forest, evaluate_tree[0],evaluate_tree[1]))

# Logistic regression method

clf_logistic = LogisticRegression().fit(X_train, Y_train)
y_pred_logistic = clf_logistic.predict(X_test)

erreur_logistic = sum(abs(y_pred_logistic-Y_test))/len(Y_test)
evaluate_logistic = evaluate_model(Y_test,y_pred_logistic)
print("Logistic Predictor : Erreur : {:.4f}, Accuracy score : {:.4f}, F-score : {:.4f}".format(erreur_logistic,evaluate_tree[0],evaluate_tree[1]))



