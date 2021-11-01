#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
sys.path.append('..')

import streamlit as st
import streamlit.components.v1 as components
import shap
import requests
import matplotlib.pyplot as plt
from json.decoder import JSONDecodeError
from api.api import get_app_details

@st.cache
def request_prediction(model_uri, customer_id):
    r = requests.post(url=model_uri, data={'sk_id_curr': customer_id})
    result = {}
    try:
        result = r.json()
        if result['status_code'] != 404:
            result['status_code'] = r.status_code
    except JSONDecodeError:
        result['status_code'] = r.status_code
    return result

API_URI = 'http://127.0.0.1:5000/predict'


# Use the full page instead of a narrow central column
st.set_page_config(layout="wide")

#shap.initjs()

def st_shap(plot, height=None):
    shap_html = f"<head>{shap.getjs()}</head><body>{plot.html()}</body>"
    components.html(shap_html, height=height)

def shap_force_plot(explainer, shap_values, Xp, feature_names):
    return shap.force_plot(explainer.expected_value[1],
                           shap_values[1][0,:], Xp[0,:],
                           feature_names=feature_names)

def shap_decision_plot(explainer, shap_values, feature_names):
    return shap.decision_plot(explainer.expected_value[1],
                              shap_values[1],
                              feature_names=feature_names)

# ## DÉBUT DU TABLEAU DE BORD ##
st.title('PretADepenser - KYC Dashboard')

st.header('Demande de prêt')
id_input = st.text_input(label='Entrez un numéro de dossier')
predict_btn = st.button('Visualiser')

if predict_btn:
    api_req = request_prediction(API_URI, id_input)
#    st.write(api_req) # debug
    if api_req['status_code'] == 404:
        st.write('La demande de prêt n\'est pas présente dans la base.')
    elif api_req['status_code'] != 200:
        st.write('Une erreur s\'est produite.')
    else:
        details = get_app_details(int(id_input))
        pred_final = api_req['risk']
        pred_label = {0: 'Acceptable', 1: 'Critique'}
        pred_proba = api_req['default_proba']
        colormap = plt.get_cmap('RdBu_r')
        color = colormap(pred_proba)

        insights = st.container()
        explain = st.container()
        
        with insights:
            st.subheader('Demande de prêt N°{}'.format(api_req['id']),
                         anchor='application')

            main_infos, prediction = st.columns([2,1])

            with main_infos:
                st.write('Coucou!')
        
            with prediction:
                st.subheader('Risque de crédit', anchor='credit-risk')
#                proba_graph = st.container()
#                pred_text = st.container()
#                with proba_graph:
                st.write('Probabilité de défaut sur le crédit demandé :')
                pie_sizes =[pred_proba, 1 - pred_proba]
                # Create a pieplot
                plt.figure(figsize=(6,6))
                plt.pie(pie_sizes, colors=[color, 'grey'])
                # add a circle at the center to transform it in a donut chart
                my_circle=plt.Circle( (0,0), 0.7, color='white')
                p=plt.gcf()
                plt.text(0,0,'{:.1f}%'.format(pred_proba * 100.0),
                            horizontalalignment='center',
                            verticalalignment='center',
                            fontsize=30)
                p.gca().add_artist(my_circle)
                st.pyplot(p)
#                with pred_text:
                st.write("Niveau de risque : **{}**".format(
                    pred_label[pred_final]))

                st.subheader('Simuler d\'autres conditions d\'emprunt')
                new_simul = st.button('Simuler')
        
        with explain:
            st.subheader('Facteurs de prédiction', anchor='predict-factors')
            st_shap(shap_force_plot(details['shap_explainer'],
                                    details['shap_values'],
                                    details['Xp'],
                                    details['feature_names']))
            for col in ['EXT_SOURCE_1','EXT_SOURCE_2', 'EXT_SOURCE_3',
                        'AMT_GOODS_PRICE', 'AMT_CREDIT']:
                if str(api_req[col]['value']) != "nan":
                    statmnt = '**{}**: {:.0f}%'.format(
                        col, abs(api_req[col]['gap_pct'] * 100.0))
                    if api_req[col]['gap_pct'] >= 0.0:
                        following = 'plus élevé que la médiane des demandes.'
                    else:
                        following = 'moins élevé que la médiane des demandes.'
                    st.write(statmnt + ' ' + following)
                else:
                    st.write('**{}**: valeur inconnue.'.format(col))

#        with customer_infos:
#            st.header('Informations sur l\'emprunteur')
