#! /usr/bin/env python3
# -*- coding: utf-8 -*-

# %%
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.preprocessing import FunctionTransformer
from sklearn.compose import make_column_transformer
from FeatureNames import get_feature_names

import warnings
warnings.simplefilter(action='ignore', category=UserWarning)


train = pd.read_csv('../02_data/application_train.csv')
#test = pd.read_csv('../02_data/application_test.csv')

# On supprime la colonne d'index et la colonne de la variable cible
train.drop(columns=['SK_ID_CURR', 'TARGET'], inplace=True)
#test.drop(columns=['SK_ID_CURR'], inplace=True)

# # Prétraitement des variables catégoriques

categor_feats = train.select_dtypes('object').columns.tolist()
# Récupération de la cardinalité des variables
dimensionality = lambda x,df : df[[x]].apply(pd.Series.nunique).values
# Division entre les catégories dites "binaires" (les flags)
# et les catégories multi dimensionnelles
categor_multid_feats = []
categor_binary_feats = []
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

#categor_multidim_preprocessor.fit_transform(train[categor_multid_featsim])
# %%

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
# ## Pipeline de prétaitement des variables catégoriques

categor_prepro = make_column_transformer(
    (categor_binary_prepro, categor_binary_feats),
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
    return categor_binary_feats + onehot_feat_names
