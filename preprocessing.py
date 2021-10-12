#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from preprocess_funcs import get_feature_names

try:
    train = pd.read_csv('02_data/application_train.csv')
except FileNotFoundError:
    try:
        train = pd.read_csv('../02_data/application_train.csv')
    except FileNotFoundError:
        raise FileNotFoundError('train data not found in the data directory')

# On supprime la colonne d'index et la colonne de la variable cible
train.drop(columns=['SK_ID_CURR', 'TARGET'], inplace=True)

# Techniques d'imputations spécifiques basées sur pandas

class CreditInfosImputer(BaseEstimator, TransformerMixin):
    '''Special missing value imputer for loan annuity and good price.
    Assigns 5% of total credit value for annuity.
    Assigns 90% of total credit value for goods price.'''
    def __init__(self):
        return None
    
    def fit(self, X, y=None):
        X.AMT_ANNUITY.fillna(round(X.AMT_CREDIT * .05, 1), inplace=True)
        X.AMT_GOODS_PRICE.fillna(round(X.AMT_CREDIT * .90, 1), inplace=True)
        return self
    
    def transform(self, X):
        X.AMT_ANNUITY.fillna(round(X.AMT_CREDIT * .05, 1), inplace=True)
        X.AMT_GOODS_PRICE.fillna(round(X.AMT_CREDIT * .90, 1), inplace=True)
        return X

credit_info_feats = ['AMT_CREDIT', 'AMT_ANNUITY', 'AMT_GOODS_PRICE']

# On assigne une valeur -1 à la variable `OWN_CAR_AGE` 
# pour les clients qui ne possèdent pas de voiture
class CarInfosImputer(BaseEstimator, TransformerMixin):
    '''Assigns a default value of -1.0 to the `OWN_CAR_AGE` variable
    for every applicant who does not own a car'''
    def __init__(self):
        return None
    
    def fit(self, X, y=None):
        mapper = {'N': 0, 'Y': 1}
        X.FLAG_OWN_CAR.replace(mapper, inplace=True)
        median_car_age = X.OWN_CAR_AGE.median()
        X.loc[X.FLAG_OWN_CAR == 0, 'OWN_CAR_AGE'] = -1.0
        X.OWN_CAR_AGE.fillna(median_car_age, inplace=True)
        return self
    
    def transform(self, X):
        mapper = {'N': 0, 'Y': 1}
        X.FLAG_OWN_CAR.replace(mapper, inplace=True)
        median_car_age = X.OWN_CAR_AGE.median()
        X.loc[X.FLAG_OWN_CAR == 0, 'OWN_CAR_AGE'] = -1.0
        X.OWN_CAR_AGE.fillna(median_car_age, inplace=True)
        return X

car_info_feats = ['FLAG_OWN_CAR', 'OWN_CAR_AGE']

# # Prétraitement des variables numériques

numeric_feats = train.select_dtypes(['int64', 'float64']).columns.tolist()

for feat in credit_info_feats + ['OWN_CAR_AGE']:
    numeric_feats.remove(feat)

flag_names = ['FLAG', 'REG_', 'LIVE']
flags = [feat for feat in numeric_feats if feat[:4] in flag_names]
categor_encoded_feats = []

# Récupération de la cardinalité des variables
dimensionality = lambda x,df : df[[x]].apply(pd.Series.nunique).values

for feat in numeric_feats:
    if feat not in flags:
        if dimensionality(feat,train) <= 2:
            categor_encoded_feats.append(feat)
            numeric_feats.remove(feat)

for flag in flags:
    categor_encoded_feats.append(flag)
    numeric_feats.remove(flag)

categor_encoded_prepro = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent'))])

numeric_avg_feats = []
numeric_med_feats = []
numeric_mod_feats = []
other_numeric_feats = []

for feat in numeric_feats:
    if feat[-4:] == '_AVG':
        numeric_avg_feats.append(feat)
    elif feat[-4:] == 'MEDI':
        numeric_med_feats.append(feat)
    elif feat[-4:] == 'MODE':
        numeric_mod_feats.append(feat)
    else:
        other_numeric_feats.append(feat)

assert len(numeric_feats) == len(numeric_avg_feats)\
                             + len(numeric_med_feats)\
                             + len(numeric_mod_feats)\
                             + len(other_numeric_feats)

# # Prétraitement des variables catégoriques

categor_feats = train.select_dtypes('object').columns.tolist()
categor_feats.remove('FLAG_OWN_CAR')

# Division entre les catégories dites "binaires" (les flags)
# et les catégories multi dimensionnelles
categor_ordinal_feats = []
categor_one_hot_feats = []
for feat in categor_feats:
    if dimensionality(feat,train) > 2:
        categor_one_hot_feats.append(feat)
    else:
        categor_ordinal_feats.append(feat)

miscategorized_feats = ['CODE_GENDER', 'WEEKDAY_APPR_PROCESS_START']
# N.B. : la variable `WEEKDAY_APPR_PROCESS_START` n'est pas binaire
# mais elle doit être traitée différemment des autres variables multi
# elle sera traitée avec les binaires pour l'instant

for feat in miscategorized_feats:
    categor_one_hot_feats.remove(feat)
    categor_ordinal_feats.append(feat)

# ## Prétraitement des variables catégoriques multidimensionnelles

def format_categor_values(x):
    y = x.lower()
    y = y.replace(' ', '_')
    y = y.replace('-', '').replace(':', '')
    y = y.replace(',', '_or').replace('/', 'or')
    return y

class OneHotColsImputer(BaseEstimator, TransformerMixin):
    def __init__(self):
        return None
    
    def fit(self, X, y=None):
        for col in X.columns.tolist():
            X[col].fillna('Unknown', inplace=True)
            X[col].replace('XNA', 'Unknown', inplace=True)
        return self
    
    def transform(self, X):
        for col in X.columns.tolist():
            X[col].fillna('Unknown', inplace=True)
            X[col].replace('XNA', 'Unknown', inplace=True)
        return X

class OneHotColsFormatter(BaseEstimator, TransformerMixin):
    def __init__(self):
        return None
    
    def fit(self, X, y=None):
        for col in X.columns.tolist():
            X[col] = X[col].apply(format_categor_values)
        return self
    
    def transform(self, X):
        for col in X.columns.tolist():
            X[col] = X[col].apply(format_categor_values)
        return X

categor_one_hot_prepro = Pipeline(steps=[
    ('imputer', OneHotColsImputer()),
    ('formatter', OneHotColsFormatter()),
    ('encoder', OneHotEncoder(handle_unknown='ignore'))])

# ## Prétraitement des variables catégoriques "binaires" (bi-dimensionnelles)

# On mappe les valeurs possibles pour chaque variable binaire
# afin de rendre l'encodage ordinal non aléatoire
# et de savoir pour chaque variable ce que représente 0 et ce que représente 1
# (ainsi pour les flags, 0 voudra toujours dire non et 1 sera toujours oui)
contract_types = ['Cash loans', 'Revolving loans']
y_or_n = ['N', 'Y']
yes_or_no = ['No', 'Yes']
genders = ['F', 'M']
weekdays = ['MONDAY', 'TUESDAY', 'WEDNESDAY', 'THURSDAY',
            'FRIDAY', 'SATURDAY', 'SUNDAY']
categories = [contract_types, y_or_n, yes_or_no, genders, weekdays]

categor_ordinal_prepro = Pipeline(steps=[
    ('nan_imputer', SimpleImputer(strategy='most_frequent')),
    ('xna_imputer', SimpleImputer(missing_values='XNA',
                                   strategy='most_frequent')),
    ('encoder', OrdinalEncoder(categories=categories))])

# # Pipeline prétraitement finale
preprocessor = ColumnTransformer([
    ('creditinfosimputer', CreditInfosImputer(), credit_info_feats),
    ('carinfosimputer', CarInfosImputer(), car_info_feats),
    ('stdnumimputer', SimpleImputer(strategy='median'), other_numeric_feats),
    ('avgimputer', SimpleImputer(strategy='mean'), numeric_avg_feats),
    ('medimputer', SimpleImputer(strategy='median'), numeric_med_feats),
    ('modimputer', SimpleImputer(strategy='most_frequent'), numeric_mod_feats),
    ('categor_ordinal', categor_ordinal_prepro, categor_ordinal_feats),
    ('categor_encoded', categor_encoded_prepro, categor_encoded_feats),
    ('categor_one_hot', categor_one_hot_prepro, categor_one_hot_feats)],
    remainder='passthrough')

def get_preprocessed_set_column_names(X):
    prepro_col_names = get_feature_names(X)
    onehot_feat_renaming = {
        k:v for k,v in zip(range(len(categor_one_hot_feats)),
                           categor_one_hot_feats)
                           }
    col_names = []
    for col_name in prepro_col_names:
        if col_name == 'TARGET':
            new_col_name = col_name
        elif col_name[:10] == 'encoder__x':
            new_col_name = col_name.replace('encoder__x', '')
            for i in range(len(categor_one_hot_feats)):
                if new_col_name[0] == str(i):
                    new_col_name = onehot_feat_renaming[i] + new_col_name[1:]
        else:
            new_col_name = col_name.split('__')[1]
        col_names.append(new_col_name)
        for i in range(len(col_names)):
            if col_names[i] == 'CODE_GENDER':
                col_names[i] = 'GENDER_male'
            elif col_names[i] == 'NAME_CONTRACT_TYPE':
                col_names[i] = 'CONTRACT_TYPE_revolving_loan'
    return col_names
