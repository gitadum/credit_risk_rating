#! /usr/bin/env python3
# -*- coding: utf-8 -*-

# %%
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.preprocessing import FunctionTransformer
from sklearn.preprocessing import MinMaxScaler
from sklearn.compose import make_column_transformer
from FeatureNames import get_feature_names

import warnings
warnings.simplefilter(action='ignore', category=UserWarning)


train = pd.read_csv('../02_data/application_train.csv')

# On supprime la colonne d'index et la colonne de la variable cible
train.drop(columns=['SK_ID_CURR', 'TARGET'], inplace=True)

# Récupération de la cardinalité des variables
dimensionality = lambda x,df : df[[x]].apply(pd.Series.nunique).values

# %%
# # Prétraitement des variables numériques
numeric_feats = train.select_dtypes(['int64', 'float64']).columns.tolist()

flag_names = ['FLAG', 'REG_', 'LIVE']
flags = [feat for feat in numeric_feats if feat[:4] in flag_names]
categor_ncoded_feats = []
for feat in numeric_feats:
    if feat not in flags:
        if dimensionality(feat,train) <= 2:
            categor_ncoded_feats.append(feat)
            numeric_feats.remove(feat)

for flag in flags:
    categor_ncoded_feats.append(flag)
    numeric_feats.remove(flag)

categor_ncoded_prepro = make_pipeline(
    (SimpleImputer(strategy='most_frequent'))
)

#print('Categorical binary:\n', categor_binary_feats)
#print('Numerical features:\n', numeric_feats)

numeric_mean_feats = []
numeric_medi_feats = []
numeric_mode_feats = []
othr_numeric_feats = []

for feat in numeric_feats:
    if feat[-4:] == '_AVG':
        numeric_mean_feats.append(feat)
    elif feat[-4:] == 'MEDI':
        numeric_medi_feats.append(feat)
    elif feat[-4:] == 'MODE':
        numeric_mode_feats.append(feat)
    else:
        othr_numeric_feats.append(feat)

assert len(numeric_feats) == len(numeric_mean_feats)\
                             + len(numeric_medi_feats)\
                             + len(numeric_mode_feats)\
                             + len(othr_numeric_feats)

numeric_imputer = make_column_transformer(
    (SimpleImputer(strategy='mean'), numeric_mean_feats),
    (SimpleImputer(strategy='median'), numeric_medi_feats),
    (SimpleImputer(strategy='most_frequent'), numeric_mode_feats),
    (SimpleImputer(strategy='median'), othr_numeric_feats),
    remainder='passthrough'
    )

numeric_prepro = make_pipeline(numeric_imputer, MinMaxScaler())
# %%
# # Prétraitement des variables catégoriques

categor_feats = train.select_dtypes('object').columns.tolist()
# Division entre les catégories dites "binaires" (les flags)
# et les catégories multi dimensionnelles
categor_binary_feats = []
categor_multid_feats = []
for feat in categor_feats:
    if dimensionality(feat,train) > 2:
        categor_multid_feats.append(feat)
    else:
        categor_binary_feats.append(feat)

miscategorized_feats = ['CODE_GENDER', 'WEEKDAY_APPR_PROCESS_START']
# N.B. : la variable `WEEKDAY_APPR_PROCESS_START` n'est pas binaire
# mais elle doit être traitée différemment des autres variables multi
# elle sera traitée avec les binaires pour l'instant

for feat in miscategorized_feats:
    categor_multid_feats.remove(feat)
    categor_binary_feats.append(feat)

# ## Prétraitement des variables catégoriques multidimensionnelles

def format_categor_values(x):
    y = x.lower()
    y = y.replace(' ', '_')
    y = y.replace('-', '').replace(':', '')
    y = y.replace(',', '_or').replace('/', 'or')
    return y

format_vfunc = np.vectorize(format_categor_values)
categor_multid_value_formatter = FunctionTransformer(lambda x: format_vfunc(x))

#concat_feat_name_with_value = lambda x: '___' + x.name + '_' + x.astype(str)

categor_multid_prepro = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='Unknown')),
    ('value_formatter', categor_multid_value_formatter),
    ('encoder', OneHotEncoder())])

#categor_multidim_preprocessor.fit_transform(train[categor_multid_feats])

# ## Prétraitement des variables catégoriques "binaires" (bi-dimensionnelles)

# On mappe les valeurs possibles pour chaque variable binaire
# afin de rendre l'encodage ordinal non aléatoire
# et de savoir pour chaque variable ce que représente 0 et ce que représente 1
# (ainsi pour les flags, 0 voudra toujours dire non et 1 sera toujours oui)
contract_types = ['Cash loans', 'Revolving loans']
y_or_n = ['N', 'Y']
yes_or_no = ['No', 'Yes']
genders = ['M', 'F']
weekdays = ['MONDAY', 'TUESDAY', 'WEDNESDAY', 'THURSDAY', 'FRIDAY', 'SATURDAY',
            'SUNDAY']
categories = [contract_types, y_or_n, y_or_n, yes_or_no, genders, weekdays]

categor_binary_prepro = Pipeline(steps=[
    ('nan_imputer', SimpleImputer(strategy='most_frequent')),
    ('xna_imputer', SimpleImputer(missing_values='XNA',
                                   strategy='most_frequent')),
    ('encoder', OrdinalEncoder(categories=categories))])

#categor_binary_preprocessor.fit_transform(train[categor_binary_feats])

# %%
# # Pipeline prétraitement finale
preprocessor = make_column_transformer(
    (numeric_prepro, numeric_feats),
    (categor_binary_prepro, categor_binary_feats),
    (categor_ncoded_prepro, categor_ncoded_feats),
    (categor_multid_prepro, categor_multid_feats),
    remainder='passthrough'
)

# %%
def get_feat_names(X):
    onehot_feat_renaming = {k:v for k,v in zip(range(len(categor_multid_feats)),
                                                     categor_multid_feats)}
    onehot_feat_names = []
    for feat_name in [n.replace('encoder__x', '')\
                      for n in get_feature_names(categor_multid_prepro.fit(
                                                 X[categor_multid_feats]))]:
        for i in range(len(categor_multid_feats)):
            if feat_name[0] == str(i):
                new_feat_name = onehot_feat_renaming[i] + feat_name[1:]
        onehot_feat_names.append(new_feat_name)
        feat_names = [
            numeric_feats +
            categor_binary_feats +
            categor_ncoded_feats +
            onehot_feat_names
        ]
    return feat_names

# %%
