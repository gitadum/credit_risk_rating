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

app = Flask(__name__)

model = joblib.load('../HomeCredit_DefaultRisk.pkl')
shap.initjs()

def final_predict(modl, X, threshold=0.5):
    return np.array(modl.predict_proba(X)[:,1] > threshold, dtype=int)

#ideal_threshold = 0.7560445445897188
ideal_threshold = 0.6672414786824148

@app.route('/', methods=['GET'])
def display_main_page():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
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
    Xp = model['p'].transform(X)
    feature_names = get_feat_names(model['p'])
    explainer = shap.TreeExplainer(model['m'])
    shap_values = explainer.shap_values(Xp)
    result['id'] = id
    result['force'] = shap.force_plot(explainer.expected_value[1],
                                      shap_values[1][0,:], Xp[0,:],
                                      feature_names=feature_names)
    result['decision'] = shap.decision_plot(explainer.expected_value[1],
                                            shap_values[1],
                                            feature_names=feature_names)
    result['status_code'] = 200
    return result



@app.route('/dashboard/')
def dashboard():
    render_template('dashboard.html')


if __name__ == '__main__':
    app.run(debug=True)
