#! /usr/bin/env python3
# -*- coding: utf-8 -*-

# %%
import pandas as pd
from app_set_preprocessing import categor_prepro, get_feat_names

train = pd.read_csv('../02_data/application_train.csv')
test = pd.read_csv('../02_data/application_test.csv')

try:
    assert len(train.SK_ID_CURR.unique()) == train.shape[0]
    assert len(test.SK_ID_CURR.unique()) == test.shape[0]
except AssertionError:
    raise AssertionError('`SK_ID_CURR` is not unic!')

train.set_index('SK_ID_CURR', inplace=True)
test.set_index('SK_ID_CURR', inplace=True)

train_encoded = categor_prepro.fit_transform(train)
#test_encoded = categor_prepro.transform(test)
# %%
print(get_feat_names(train))
