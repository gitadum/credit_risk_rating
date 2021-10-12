#! /usr/bin/env python3
# -*- coding: utf-8 -*-
#%%
import streamlit as st
import requests

def request_prediction(model_uri, customer_id):
    r = requests.post(url=model_uri, data={'sk_id_curr': customer_id})
    
    if r.status_code != 200:
        raise Exception(
            "Request failed with status {}, {}".format(r.status_code, r.text))
    
    return r.json()

API_URI = 'http://127.0.0.1:5000'

# %%

st.title('PretADepenser - KYC Dashboard')

st.header('Identité du client')

id_input = st.text_input(label='Entrez un ID Client')
predict_btn = st.button('Visualiser')

if predict_btn:
    model_prediction = request_prediction(API_URI, id_input)
    pred_final = model_prediction['predict_final']
    pred_final_fr = {'Favorable': 'Acceptable', 'Unfavorable': 'Critique'}
    st.header('Modélisation du risque de crédit :')
    st.write("Niveau de risque : **{}**".format(pred_final_fr[pred_final]))
    st.write('**{}%** de chances d\'aboutir à un défaut de paiement.'.format(
        round(model_prediction['predict_proba'],1) * 100.0
    ))

st.header('Simuler d\'autres conditions d\'emprunt')

new_simul = st.button('Simuler')