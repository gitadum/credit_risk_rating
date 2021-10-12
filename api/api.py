#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
sys.path.append('..')
import joblib
import pandas as pd
import numpy as np
from flask import Flask, jsonify, request, render_template

app = Flask(__name__)


model = joblib.load('../HomeCredit_DefaultRisk.pkl')

def final_predict(modl, X, threshold=0.5):
    return np.array(modl.predict_proba(X)[:,1] > threshold, dtype=int)

#ideal_threshold = 0.7560445445897188
ideal_threshold = 0.6672414786824148

@app.route('/', methods=['GET'])
def display_main_page():
    return render_template('index.html')

@app.route('/', methods=['POST'])
def predict():
    client_db = pd.read_csv('../02_data/application_test.csv', index_col=0)
    id = int(request.form['sk_id_curr'])
    client_vals = client_db.loc[id, :]
    X = pd.DataFrame(client_vals.values.reshape(1,-1),columns=client_db.columns)
    for k,v in client_db.dtypes.items():
        X[k] = X[k].astype(v)
    result = {}
    result['id'] = id
    result['predict_proba'] = model.predict_proba(X)[:,1][0]
    prediction = final_predict(model, X, threshold=ideal_threshold)[0]
    if prediction == 0:
        result['predict_final'] = 'Favorable'
    elif prediction == 1:
        result['predict_final'] = 'Unfavorable'
    return jsonify(result)

@app.route('/dashboard/')
def dashboard():
    render_template('dashboard.html')


if __name__ == '__main__':
    app.run(debug=True)
