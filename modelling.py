#! /usr/bin/env python3
# -*- coding: utf-8 -*-

# %%
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestClassifier
from preprocessing import preprocessor, get_preprocessed_set_column_names

#%%
train = pd.read_csv('02_data/application_train.csv')
test = pd.read_csv('02_data/application_test.csv')

id_error_msg = lambda x: '`SK_ID_CURR` is not unic for {} set!'.format(x)
assert len(train.SK_ID_CURR.unique()) == train.shape[0], id_error_msg('train')
assert len(test.SK_ID_CURR.unique()) == test.shape[0], id_error_msg('test')
train.set_index('SK_ID_CURR', inplace=True)
test.set_index('SK_ID_CURR', inplace=True)

X, y = train.iloc[:, 1:], train.iloc[:, 0]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2)
# %%
X_transformed = preprocessor.fit_transform(X)
#test_preprocessed = preprocessor.transform(test)

print(X_transformed.shape)
# %%
df = pd.DataFrame(X_transformed,
                  columns=get_preprocessed_set_column_names(preprocessor))
# %%
model = make_pipeline(preprocessor, RandomForestClassifier())
model.fit(X_train, y_train)
# %%
model.get_params()

# %%
y_pred = model.predict(X_test)
# %%
