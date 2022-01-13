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
from preprocessing import add_secondary_table_features
from preprocessing import categor_encoded_feats
from modelling import final_predict

CONTEXT = 'heroku'

if CONTEXT == 'local':
    HOST_URL = 'https://127.0.0.1'
elif CONTEXT == 'heroku':
    HOST_URL = 'https://powerful-tor-37001.herokuapp.com'
PORT = 5000

app = Flask(__name__)

app_db = load_dataset('application_test.csv', index_col=0)
app_db = add_secondary_table_features(app_db)

model = load_model('HomeCredit_DefaultRisk.pkl')

ideal_threshold = 0.6757005832464233

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
                if col not in categor_encoded_feats:
                    result[col]['median'] = float(d['app_db'][col].median())
                    result[col]['mean'] = float(d['app_db'][col].mean())
    
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

def gap_with_trends(prediction):
    
    def central_trend_gap(value, trend):
        result = {}
        result['gap'] = value - trend
        if trend != 0.0:
            result['gap_pct'] = result['gap'] / trend
        else:
            result['gap_pct'] = np.nan
        return result
    
    result = {}
    for col in prediction.keys():
        if type(prediction[col]) == dict:
            try:
                avg = central_trend_gap(prediction[col]['value'],
                                        prediction[col]['mean'])
                med = central_trend_gap(prediction[col]['value'],
                                        prediction[col]['median'])
                #print(avg) # debug
                #print(med) # debug
                result[col] = {}
                result[col]['gap_avg'] = avg['gap']
                result[col]['gap_avg_pct'] = avg['gap_pct']
                result[col]['gap_med'] = med['gap']
                result[col]['gap_med_pct'] = med['gap_pct']
                #print(result) # debug
            except KeyError:
                continue
        else:
            continue
    return result

if __name__ == '__main__':
    app.run()