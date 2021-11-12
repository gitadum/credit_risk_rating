#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import sys

sys.path.append('.')
sys.path.append('..')

import pandas as pd
import numpy as np
import shap
from flask import Flask, jsonify, request
from load_files import load_dataset, load_model
from preprocessing import get_preprocessed_set_column_names as get_feat_names
from preprocess_funcs import add_secondary_table_features
from timer import timer

app = Flask(__name__)

app_db = load_dataset('application_test.csv', index_col=0)
app_db = add_secondary_table_features(app_db)

model = load_model('HomeCredit_DefaultRisk.pkl')

def final_predict(modl, X, threshold=0.5):
    return np.array(modl.predict_proba(X)[:,1] > threshold, dtype=int)

#ideal_threshold = 0.7560445445897188
#ideal_threshold = 0.6672414786824148
ideal_threshold = 0.6757005832464233

#@app.route('/app_details', methods=['POST'])
@timer
def get_app_details(id):
    result = {}
    try:
        client_vals = app_db.loc[id, :]
    except KeyError:
        result['status_code'] = 404
        return result
    X = pd.DataFrame(client_vals.values.reshape(1,-1),columns=app_db.columns)
    for k,v in app_db.dtypes.items():
        X[k] = X[k].astype(v)
    result['id'] = id
    result['app_db'] = app_db
    Xp = model['p'].transform(X)
    result['X'] = X
    result['Xp'] = Xp
    result['feature_names'] = get_feat_names(model['p'])
    explainer = shap.TreeExplainer(model['m'])
    shap_values = explainer.shap_values(Xp)
    result['shap_explainer'] = explainer
    result['shap_values'] = shap_values
    result['status_code'] = 200
    return result

@app.route('/predict', methods=['POST'])
@timer
def predict():
    id = int(request.form['sk_id_curr'])
    d = get_app_details(id)
    result = {}
    try:
        assert d['status_code'] == 200
    except AssertionError:
        result['status_code'] = d['status_code']
        return result
    result['id'] = d['id']
    result['default_proba'] = float(model.predict_proba(d['X'])[:,1][0])
    result['risk'] = int(final_predict(model, d['X'], ideal_threshold)[0])

    for col in list(d['app_db'].columns):
        result[col] = dict()
        result[col]['value'] = d['app_db'].loc[id, col]
        if d['app_db'][col].dtype in ['float64', 'int64']:
            if type(result[col]['value']) == np.int64:
                result[col]['value'] = int(result[col]['value'])
            elif type(result[col]['value']) == np.float64:
                result[col]['value'] = float(result[col]['value'])
            result[col]['median'] = float(d['app_db'][col].median())
            result[col]['gap'] = result[col]['value'] - result[col]['median']
            if result[col]['median'] != 0.0:
                result[col]['gap_pct'] = result[col]['gap'] \
                                       / result[col]['median']
            else:
                result[col]['gap_pct'] = np.nan
    
    result['status_code'] = 200
    
    # debug: check type of result entries
    # numpy arrays are not jsonifiable
    # for key in result:
    #     if type(result[key]) == dict:
    #         for k,v in result[key].items():
    #             print(key, k, type(v))
    #     else:
    #         print(key, type(result[key]))
    
    return jsonify(result)


if __name__ == '__main__':
    app.run(debug=True)
