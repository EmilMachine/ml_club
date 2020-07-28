#!/usr/bin/env python
# coding: utf-8

# # packages used

# misc
import numpy as np

# DATA - prep
#kaggle
import pandas as pd
import sklearn.model_selection

# ML - models 
import sklearn.linear_model
import sklearn.tree
import sklearn.ensemble
import xgboost

# hypertune
from sklearn.model_selection import GridSearchCV

# ML - accuracy
import sklearn.metrics

# Plot and visualize
# import matplotlib.pyplot as plt
# import shap

import joblib

# # Get data

# Setup:
# - follow "API credential step" listed here: https://github.com/Kaggle/kaggle-api
#     - go to https://www.kaggle.com/ (login)
#     - go to my_profile (download kaggle.json)
#     - put it in ~/.kaggle/kaggle.json
#     - `cp ~/Downloads/kaggle.json ~/.kaggle/kaggle.json`
#     - `chmod 600 ~/.kaggle/kaggle.json`
# - Go to kaggle and join competition: 
#     - https://www.kaggle.com/c/titanic
# - install kaggle
# - download data
# - profit!!!

metadata = {
    'basepath' : 'data/',
    'dataset':'titanic',
    'train' : 'train.csv',
    'test' : 'test.csv'}

# get data
# make folder
# download .zip
# unzip
# remove the .zip
# (data is placed ../data/titanic)

# !mkdir -p {metadata['basepath']}
# !kaggle competitions download -c dataset {metadata['dataset']} -p {metadata['basepath']}
# !unzip -o {metadata['basepath']}{metadata['dataset']}.zip -d {metadata['basepath']}{metadata['dataset']}/
# !rm {metadata['basepath']}{metadata['dataset']}.zip


# # Load and explore

# load
train = pd.read_csv("{basepath}/{dataset}/{train}".format(**metadata))
test = pd.read_csv("{basepath}/{dataset}/{test}".format(**metadata))


# # Simple imputation + cleaning
def clean(df):
    dfc = df.copy()
    
    # Simple map
    dfc['Sex'] = dfc['Sex'].map({"female":0,"male":1}).astype(int)
    
    # simple Impute
    dfc['Age'] = dfc["Age"].fillna(-1)
    dfc['Fare'] = dfc["Fare"].fillna(-1)

    # Simple feature engineering (combining two variables)
    dfc['FamilySize'] = dfc['SibSp'] + dfc['Parch'] + 1

    # Simple feature engineering (converting to boolean)
    dfc['Has_Cabin'] = dfc["Cabin"].apply(lambda x: 0 if type(x) == float else 1)
    dfc = dfc.drop(["Cabin"],axis=1)
    
    # "Stupid feature engineering - apply length 
    dfc['Name_length'] = dfc['Name'].apply(len)
    dfc = dfc.drop(["Name"],axis=1)
    dfc['Ticket_length'] = dfc['Ticket'].apply(len)
    dfc = dfc.drop(["Ticket"],axis=1)
    
    # 1-hot encoding - different options are encoded as booleans
    # ie. 1 categorical - become 3: 0-1 features.
    dfc['Embarked_Q'] = dfc['Embarked'].apply(lambda x: 1 if x=="Q" else 0)
    dfc['Embarked_S'] = dfc['Embarked'].apply(lambda x: 1 if x=="S" else 0)
    dfc['Embarked_C'] = dfc['Embarked'].apply(lambda x: 1 if x=="C" else 0)
    dfc = dfc.drop(["Embarked"],axis=1)
    
    
    return dfc


clean_train = clean(train)
clean_test = clean(test)

target = "Survived"

y = clean_train[target]
X = clean_train.drop([target],axis=1)

# Split data in train and validation
seed = 42
test_size = 0.7

X_train, X_val, y_train, y_val = sklearn.model_selection.train_test_split(
    X,
    y,
    random_state = seed,
    test_size = test_size)


# # ML


# xgboost
model_xgboost = xgboost.sklearn.XGBClassifier()
model_xgboost.fit(X_train, y_train);


# # Hyper parameter tuning

# ## xgboost

##### from sklearn.model_selection import GridSearchCV

xgb_param_grid = {
        'n_estimators' : [100,200,300],
        'max_depth': [2,3,5,8],
        'min_sample_split': [2,3,5,8],
        "learning_rate": [0.05,0.1,0.5]
}


model_xgboost_dummy = xgboost.sklearn.XGBClassifier(min_sample_split=3,learning_rate=0.1)
grid_search_xgb = GridSearchCV(estimator = model_xgboost_dummy, param_grid = xgb_param_grid , 
                          cv = 5, n_jobs = -1, verbose = 2)

grid_search_xgb.fit(X_train, y_train)

best_grid = grid_search_xgb.best_params_
model_xgboost_best = xgboost.sklearn.XGBClassifier(**best_grid)
model_xgboost_best.fit(X_train,y_train);


# # Eval ML

models = {
    "model_xgboost_best":model_xgboost_best
}

for name,model in zip(models.keys(),models.values()):
    acc = sklearn.metrics.accuracy_score(
     y_true = y_val,
     y_pred = model.predict(X_val)
    )
    acc_train = sklearn.metrics.accuracy_score(
     y_true = y_train,
     y_pred = model.predict(X_train)
    )
    
    print(name," | train: ",round(acc_train,3)," | test: ",round(acc,3))

# # save the model

# save the model to disk


filename =  metadata["basepath"] + 'model_xgboost_best.sav'
joblib.dump(model_xgboost_best, filename)
 
