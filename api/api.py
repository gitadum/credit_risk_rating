#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
sys.path.append('..')
import joblib
import pandas as pd
import numpy as np
import shap
from flask import Flask, jsonify, request, render_template
from preprocessing import get_preprocessed_set_column_names as get_feat_names
from preprocess_funcs import add_secondary_table_features
from timer import timer

app = Flask(__name__)

model = joblib.load('../HomeCredit_DefaultRisk.2.pkl')
shap.initjs()

def final_predict(modl, X, threshold=0.5):
    return np.array(modl.predict_proba(X)[:,1] > threshold, dtype=int)

#ideal_threshold = 0.7560445445897188
#ideal_threshold = 0.6672414786824148
ideal_threshold = 0.6757005832464233

@app.route('/', methods=['GET'])
def display_main_page():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
@timer
def predict():
    client_db = pd.read_csv('../02_data/application_test.csv', index_col=0)
    id = int(request.form['sk_id_curr'])
    result = {}
    try:
        client_vals = client_db.loc[id, :]
    except KeyError:
        result['status_code'] = 404
        return jsonify(result)
    X = pd.DataFrame(client_vals.values.reshape(1,-1),columns=client_db.columns)
    for k,v in client_db.dtypes.items():
        X[k] = X[k].astype(v)
    X = add_secondary_table_features(X)
    result['id'] = id
    result['predict_proba'] = model.predict_proba(X)[:,1][0]
    prediction = final_predict(model, X, threshold=ideal_threshold)[0]
    if prediction == 0:
        result['predict_final'] = 'Favorable'
    elif prediction == 1:
        result['predict_final'] = 'Unfavorable'
    result['status_code'] = 200
    return jsonify(result)

def explain_prediction(id):
    client_db = pd.read_csv('../02_data/application_test.csv', index_col=0)
    result = {}
    try:
        client_vals = client_db.loc[id, :]
    except KeyError:
        result['status_code'] = 404
        return jsonify(result)
    X = pd.DataFrame(client_vals.values.reshape(1,-1),columns=client_db.columns)
    for k,v in client_db.dtypes.items():
        X[k] = X[k].astype(v)
    X = add_secondary_table_features(X)
    Xp = model['p'].transform(X)
    feature_names = get_feat_names(model['p'])
    explainer = shap.TreeExplainer(model['m'])
    shap_values = explainer.shap_values(Xp)
    result['id'] = id
    force_plot = shap.force_plot(explainer.expected_value[1],
                                      shap_values[1][0,:], Xp[0,:],
                                      feature_names=feature_names,)
#                                      matplotlib=True)
#    decision_plot = shap.decision_plot(explainer.expected_value[1],
#                                            shap_values[1],
#                                            feature_names=feature_names)
#    result['decision'] = decision_plot
    result['force'] = force_plot
    for col in ['EXT_SOURCE_1','EXT_SOURCE_2', 'EXT_SOURCE_3',
                'AMT_GOODS_PRICE', 'AMT_CREDIT']:
        result[col] = dict()
        result[col]['value'] = client_db.loc[id, col]
        result[col]['median'] = client_db[col].median()
        result[col]['gap'] = result[col]['value'] - result[col]['median']
        result[col]['gap_pct'] = result[col]['gap'] / result[col]['median']
    result['status_code'] = 200
    return result



@app.route('/dashboard/')
def dashboard():
    render_template('dashboard.html')


if __name__ == '__main__':
    app.run(debug=True)
