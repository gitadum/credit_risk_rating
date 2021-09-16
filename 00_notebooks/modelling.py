#! /usr/bin/env python3
# -*- coding: utf-8 -*-

# %%
import pandas as pd
from preprocessing import preprocessor, get_preprocessed_set_column_names

train = pd.read_csv('../02_data/application_train.csv')
test = pd.read_csv('../02_data/application_test.csv')

try:
    assert len(train.SK_ID_CURR.unique()) == train.shape[0]
    assert len(test.SK_ID_CURR.unique()) == test.shape[0]
except AssertionError:
    raise AssertionError('`SK_ID_CURR` is not unic!')

train.set_index('SK_ID_CURR', inplace=True)
test.set_index('SK_ID_CURR', inplace=True)

train_preprocessed = preprocessor.fit_transform(train)
#test_preprocessed = preprocessor.transform(test)

print(train_preprocessed.shape)
# %%
df = pd.DataFrame(train_preprocessed,
                  columns=get_preprocessed_set_column_names(preprocessor))
# %%
