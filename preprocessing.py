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
from load_files import load_dataset

train = load_dataset('application_train.csv')

# On supprime la colonne d'index et la colonne de la variable cible
train.drop(columns=['SK_ID_CURR', 'TARGET'], inplace=True)

# Listes de colonnes qui vont passer par une chaîne de prétraitements spéciale
age_info_feats = ['DAYS_BIRTH']
credit_info_feats = ['AMT_CREDIT', 'AMT_ANNUITY', 'AMT_GOODS_PRICE']
car_info_feats = ['FLAG_OWN_CAR', 'OWN_CAR_AGE']

# # Prétraitement des variables numériques

numeric_feats = train.select_dtypes(['int64', 'float64']).columns.tolist()

for feat in age_info_feats + credit_info_feats + car_info_feats:
    if feat in numeric_feats:
        numeric_feats.remove(feat)

# Fonction qui récupère la cardinalité d'une variable
dimensionality = lambda x,df : df[[x]].apply(pd.Series.nunique).values

flags = [feat for feat in numeric_feats if feat[:4] in ['FLAG', 'REG_', 'LIVE']]
categor_encoded_feats = []
for feat in numeric_feats:
    if feat not in flags:
        if dimensionality(feat,train) <= 2:
            categor_encoded_feats.append(feat)
            numeric_feats.remove(feat)

for flag in flags:
    categor_encoded_feats.append(flag)
    numeric_feats.remove(flag)

numeric_avg_feats = []
numeric_med_feats = []
numeric_mod_feats = []
numeric_notcntral = []
for feat in numeric_feats:
    if feat[-4:] == '_AVG':
        numeric_avg_feats.append(feat)
    elif feat[-4:] == 'MEDI':
        numeric_med_feats.append(feat)
    elif feat[-4:] == 'MODE':
        numeric_mod_feats.append(feat)
    else:
        numeric_notcntral.append(feat)

assert len(numeric_feats) == len(numeric_avg_feats)\
                           + len(numeric_med_feats)\
                           + len(numeric_mod_feats)\
                           + len(numeric_notcntral)

# # Prétraitement des variables catégoriques

categor_feats = train.select_dtypes('object').columns.tolist()

for feat in age_info_feats + credit_info_feats + car_info_feats:
    if feat in categor_feats:
        categor_feats.remove(feat)

# Division entre les catégories dites "binaires" (les flags)
# qui seront encodées de manière ordinale
# et les catégories multi-dimensionnelles
# qui seront traitées avec un encodeur one-hot
categor_ordinal_feats = []
categor_one_hot_feats = []
for feat in categor_feats:
    if dimensionality(feat,train) > 2:
        categor_one_hot_feats.append(feat)
    else:
        categor_ordinal_feats.append(feat)

# N.B. : la variable `WEEKDAY_APPR_PROCESS_START` n'est pas binaire
# mais elle doit être encodée de manière ordinale de par sa nature cyclique
for feat in ['CODE_GENDER', 'WEEKDAY_APPR_PROCESS_START']:
    categor_one_hot_feats.remove(feat)
    categor_ordinal_feats.append(feat)

categor_encoded_prepro = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent'))])

# ## Prétraitement des variables catégoriques multidimensionnelles

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
    
    def format_categor_values(x):
        y = x.lower()
        y = y.replace(' ', '_')
        y = y.replace('-', '').replace(':', '')
        y = y.replace(',', '_or').replace('/', 'or')
        return y
    
    def fit(self, X, y=None):
        for col in X.columns.tolist():
            X[col] = X[col].apply(OneHotColsFormatter.format_categor_values)
        return self
    
    def transform(self, X):
        for col in X.columns.tolist():
            X[col] = X[col].apply(OneHotColsFormatter.format_categor_values)
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
ordinal_dims = {'contract_types': ['Cash loans', 'Revolving loans'],
                'y_or_n': ['N', 'Y'],
                'yes_or_no': ['No', 'Yes'],
                'genders': ['F', 'M'],
                'weekdays': ['MONDAY', 'TUESDAY', 'WEDNESDAY', 'THURSDAY',
                             'FRIDAY', 'SATURDAY', 'SUNDAY']
                 }
ordinal_categories = [ordinal_dims[k] for k in ordinal_dims.keys()]

categor_ordinal_prepro = Pipeline(steps=[
    ('nan_imputer', SimpleImputer(strategy='most_frequent')),
    ('xna_imputer', SimpleImputer(missing_values='XNA',
                                   strategy='most_frequent')),
    ('encoder', OrdinalEncoder(categories=ordinal_categories))])

# Techniques d'imputations spécifiques basées sur pandas

class AgeInfosTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        return None
    
    def fit(self, X, y=None):
        X.loc[:, 'YEARS_AGE'] = - X.loc[:, 'DAYS_BIRTH'] / 365.0
        X.drop(columns=['DAYS_BIRTH'])
        return self
    
    def transform(self, X):
        X.loc[:, 'YEARS_AGE'] = - X.loc[:, 'DAYS_BIRTH'] / 365.0
        X.drop(columns=['DAYS_BIRTH'], inplace=True)
        return X

class CreditInfosImputer(BaseEstimator, TransformerMixin):
    '''Special missing value imputer for loan annuity and good price.
    Assigns 5% of total credit value for annuity.
    Assigns 90% of total credit value for goods price.'''
    def __init__(self):
        return None
    
    def fit(self, X, y=None):
        X.AMT_ANNUITY.fillna(round(X.AMT_CREDIT * .05, 1), inplace=True)
        X.AMT_GOODS_PRICE.fillna(round(X.AMT_CREDIT * .90, 1), inplace=True)
        X['CREDIT_TERM'] = X.AMT_ANNUITY / X.AMT_CREDIT
        return self
    
    def transform(self, X):
        X.AMT_ANNUITY.fillna(round(X.AMT_CREDIT * .05, 1), inplace=True)
        X.AMT_GOODS_PRICE.fillna(round(X.AMT_CREDIT * .90, 1), inplace=True)
        X['CREDIT_TERM'] = X.AMT_ANNUITY / X.AMT_CREDIT
        return X

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

# # Pipeline prétraitement finale
preprocessor = ColumnTransformer([
    ('ageinfostransformer', AgeInfosTransformer(), age_info_feats),
    ('creditinfosimputer', CreditInfosImputer(), credit_info_feats),
    ('carinfosimputer', CarInfosImputer(), car_info_feats),
    ('numimputer', SimpleImputer(strategy='median'), numeric_notcntral),
    ('avgimputer', SimpleImputer(strategy='mean'), numeric_avg_feats),
    ('medimputer', SimpleImputer(strategy='median'), numeric_med_feats),
    ('modimputer', SimpleImputer(strategy='most_frequent'), numeric_mod_feats),
    ('categor_ordinal', categor_ordinal_prepro, categor_ordinal_feats),
    ('categor_encoded', categor_encoded_prepro, categor_encoded_feats),
    ('categor_one_hot', categor_one_hot_prepro, categor_one_hot_feats)],
    remainder='passthrough')

def get_preprocessed_set_column_names(prepro):
    prepro_col_names = get_feature_names(prepro)

    remainder_cols = []
    for f in prepro.feature_names_in_:
        if f not in [item for sublist in prepro._columns for item in sublist]:
            remainder_cols.append(f)
    
    onehot_feat_cols = prepro.transformers_[-2][2]
    onehot_feat_renaming = {k:v for k,v in zip(range(len(onehot_feat_cols)),
                                               onehot_feat_cols)}
    
    col_names = []
    k = 0
    for col_name in prepro_col_names:
        if col_name[:10] == 'encoder__x':
            new_col_name = col_name.replace('encoder__x', '')
            for i in range(len(onehot_feat_cols)):
                if new_col_name[0] == str(i):
                    new_col_name = onehot_feat_renaming[i] + new_col_name[1:]
        elif col_name[0] == 'x' and float(col_name[1:]).is_integer() is True:
            new_col_name = remainder_cols[k]
            k += 1
        else:
            new_col_name = col_name.split('__')[1]
        col_names.append(new_col_name)
        for n in range(len(col_names)):
            if col_names[n] == 'DAYS_BIRTH':
                col_names[n] = 'YEARS_AGE'
            elif col_names[n] == 'CODE_GENDER':
                col_names[n] = 'GENDER_male'
            elif col_names[n] == 'NAME_CONTRACT_TYPE':
                col_names[n] = 'CONTRACT_TYPE_revolving_loan'
    col_names.insert(4, 'CREDIT_TERM')
    return col_names

def add_secondary_table_features(df):
    df = df.copy()
    bur = load_dataset('bureau.csv')
    idx = 'SK_ID_CURR'
    days_before_curr_app_mean = np.abs(bur.groupby(idx).DAYS_CREDIT.mean())
    df['bureau_DAYS_CREDIT_mean'] = df.join(
        days_before_curr_app_mean, how='left').DAYS_CREDIT
    df['bureau_DAYS_CREDIT_mean'].fillna(- 1.0, inplace=True)
    days_before_curr_app_min = np.abs(bur.groupby(idx).DAYS_CREDIT.max())
    df['bureau_DAYS_CREDIT_min'] = df.join(
        days_before_curr_app_min, how='left').DAYS_CREDIT
    df['bureau_DAYS_CREDIT_min'].fillna(- 1.0, inplace=True)
    n_active_credits = bur[bur.CREDIT_ACTIVE == 'Active'].groupby(idx)\
                                                         .CREDIT_ACTIVE.count()
    df['bureau_CREDIT_ACTIVE_count'] = df.join(
        n_active_credits, how='left').CREDIT_ACTIVE
    df.bureau_CREDIT_ACTIVE_count.fillna(0, inplace=True)
    mean_days_cred_overdue = bur.groupby(idx).CREDIT_DAY_OVERDUE.mean()
    df['bureau_CREDIT_DAY_OVERDUE_mean'] = df.join(
        mean_days_cred_overdue, how='left').CREDIT_DAY_OVERDUE
    df['bureau_CREDIT_DAY_OVERDUE_mean'].fillna(0, inplace=True)
    max_enddate = bur.groupby(idx).DAYS_CREDIT_ENDDATE.max()
    df['bureau_DAYS_CREDIT_ENDDATE_max'] = df.join(
        max_enddate, how='left').DAYS_CREDIT_ENDDATE
    df.bureau_DAYS_CREDIT_ENDDATE_max.fillna(0, inplace=True)
    return df
