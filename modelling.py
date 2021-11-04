#! /usr/bin/env python3
# -*- coding: utf-8 -*-

# Importations
import joblib

# Bibliothèques utiles
import pandas as pd

# Prétraitements
from preprocessing import preprocessor
from imblearn.pipeline import Pipeline
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import train_test_split

# Machine Learning
from lightgbm import LGBMClassifier
from modelling_funcs import model_eval
#from sklearn.metrics import recall_score, precision_score

#from styles import *

# Chargement des jeux de données d'apprentissage
train = pd.read_csv('02_data/application_train.csv', index_col=0)
test = pd.read_csv('02_data/application_test.csv', index_col=0)

print('Training set dimensions :', train.shape)
df = train.copy()

cls_size = df.TARGET.value_counts()
cls_freq = df.TARGET.value_counts(normalize=True)
print(pd.DataFrame({'size': cls_size,
                    'freq': cls_freq.apply(lambda x: '%.2f' % x)}))

X, y = train.iloc[:, 1:], train.iloc[:, 0]

# Séparation du jeu de données entre entraînement et évaluation
#r = 42
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2)
#                                                   random_state=r)

# Définition de la pipeline de modélisation finale
undersampler = RandomUnderSampler()

best_model_params = {'boosting_type': 'gbdt',
                     'class_weight': None,
                     'colsample_bytree': 0.8736655622105718,
                     'importance_type': 'split',
                     'learning_rate': 0.1,
                     'max_depth': -1,
                     'min_child_samples': 80,
                     'min_child_weight': 100.0,
                     'min_split_gain': 0.0,
                     'n_estimators': 373,
                     'n_jobs': -1,
                     'num_leaves': 12,
                     'objective': None,
                     'random_state': None,
                     'reg_alpha': 16,
                     'reg_lambda': 21,
                     'silent': True,
                     'subsample': 0.7981160065359487,
                     'subsample_for_bin': 200000,
                     'subsample_freq': 0}

model = Pipeline([('u', undersampler),
                  ('p', preprocessor),
                  ('m', LGBMClassifier(**best_model_params))])

# Entraînement du modèle final
model.fit(X_train, y_train)

# Évaluation du modèle final
model_eval(model, X_test, y_test)

# Sérialisation du modèle final avec joblib
joblib.dump(model, 'model/HomeCredit_DefaultRisk.pkl')

# Chargement du modèle sérialisé
loaded_model = joblib.load('model/HomeCredit_DefaultRisk.pkl')
loaded_model.predict(X_test)
loaded_model.predict_proba(X_test)
