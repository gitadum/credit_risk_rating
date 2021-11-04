#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd

# Chargement des jeux de données d'apprentissage et de test
try:
    app_train = pd.read_csv('02_data/application_train.csv', index_col=0)
    app_test = pd.read_csv('02_data/application_test.csv', index_col=0)
except FileNotFoundError:
    app_train = pd.read_csv('../02_data/application_train.csv', index_col=0)
    app_test = pd.read_csv('../02_data/application_test.csv', index_col=0)
