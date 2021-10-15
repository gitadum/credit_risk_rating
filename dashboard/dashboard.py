#! /usr/bin/env python3
# -*- coding: utf-8 -*-
#%%
import streamlit as st
import requests
import matplotlib.pyplot as plt

def request_prediction(model_uri, customer_id):
    r = requests.post(url=model_uri, data={'sk_id_curr': customer_id})
    
    if r.status_code != 200:
        raise Exception(
            "Request failed with status {}, {}".format(r.status_code, r.text))
    
    return r.json()

API_URI = 'http://127.0.0.1:5000/predict'

# %%

st.title('PretADepenser - KYC Dashboard')

st.header('Identité du client')

id_input = st.text_input(label='Entrez un ID Client')
predict_btn = st.button('Visualiser')

if predict_btn:
    model_prediction = request_prediction(API_URI, id_input)
    pred_final = model_prediction['predict_final']
    pred_final_fr = {'Favorable': 'Acceptable', 'Unfavorable': 'Critique'}
    pred_proba = model_prediction['predict_proba']
    prediction, explanation, customer_infos = st.columns(3)
    
    with prediction:
        st.header('Modélisation du risque de crédit :')
        proba_graph = st.container()
        pred_text = st.container()
        with proba_graph:
            st.write('probabilité de défaut sur le crédit demandé :')
#            plt.draw()
            pie_sizes =[pred_proba, 1 - pred_proba]
            # Create a pieplot
            plt.pie(pie_sizes, colors=['blue', 'grey'])
            # add a circle at the center to transform it in a donut chart
            my_circle=plt.Circle( (0,0), 0.7, color='white')
            p=plt.gcf()
            plt.text(0,0,'{:.1f}%'.format(pred_proba * 100.0),
                        horizontalalignment='center',
                        verticalalignment='center',
                        fontsize=30)
            p.gca().add_artist(my_circle)
            st.pyplot(p)
        with pred_text:
            st.write("Niveau de risque : **{}**".format(pred_final_fr[pred_final]))
#            st.write('**{:.1f}%** de chances d\'aboutir à un défaut de paiement.'\
#                .format(pred_proba * 100.0))

        st.header('Simuler d\'autres conditions d\'emprunt')
        new_simul = st.button('Simuler')
    
    with explanation:
        st.header('Facteurs de prédiction')
#       st.pyplot(customer_explain['force'])

    with customer_infos:
        st.header('Informations sur le client')