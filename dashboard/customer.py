#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
sys.path.append('.')
sys.path.append('..')

import streamlit as st
import streamlit.components.v1 as components
import shap
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

import requests
from json import JSONDecodeError
from api.app import get_app_details, gap_with_trends

from math import floor

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

@st.cache
def cached_app_details():
    return get_app_details


API_URI = 'http://127.0.0.1:5000/predict'

# Use the full page instead of a narrow central column
st.set_page_config(layout="wide")

#shap.initjs()
def st_shap(plot, height=185, width=None):
    shap_html = f"<head>{shap.getjs()}</head><body style=\"width: 3000px;\">\
                  {plot.html()}</body>"
    components.html(shap_html, height=height, width=width, scrolling=True)

def shap_force_plot(explainer, shap_values, Xp, feature_names):
    return shap.force_plot(explainer.expected_value[1],
                           shap_values[1][0,:], Xp[0,:],
                           feature_names=feature_names,
                           contribution_threshold=0.05)

def shap_decision_plot(explainer, shap_values, feature_names):
    return shap.decision_plot(explainer.expected_value[1],
                              shap_values[1],
                              feature_names=feature_names)

#@st.cache(suppress_st_warning=True)
def customer_dashboard():
    # ## DÉBUT DU TABLEAU DE BORD ##
    #st.title('PretADepenser - KYC Dashboard', anchor='/')

    id_input = st.text_input(label='Entrez un numéro de dossier')
    predict_btn = st.button('Visualiser')

    if predict_btn:
        req = request_prediction(API_URI, id_input)
        gaps_req = gap_with_trends(req)
        st.write(req) # debug
        # for col in list(req.keys()):
        #     print(col, type(col))
        st.write(gaps_req) # debug
        if req['status_code'] == 404:
            st.write('La demande de prêt n\'est pas présente dans la base.')
        elif req['status_code'] != 200:
            st.write('Une erreur s\'est produite.')
        else:
            details = get_app_details(int(id_input))
            pred_final = req['risk']
            pred_label = {0: 'Acceptable', 1: 'Critique'}
            pred_proba = req['default_proba']
            red =  '#FF0D57'
            blue = '#1E88E5'
            shap_cm = LinearSegmentedColormap.from_list("", [blue,'violet',red])
            color = shap_cm(pred_proba)

            insights = st.container()
            explain = st.container()
            
            with insights:
                st.header('Demande de prêt N°{}'.format(req['id']),
                            anchor='application')

                prediction, explain = st.columns([1,2])

                with prediction:
                    st.subheader('Risque de crédit', anchor='credit-risk')
                    st.write("Niveau de risque : **{}**".format(
                        pred_label[pred_final]))
                    st.write("[que signifient ces seuils de risque ?](#model)")
                    st.write('Probabilité de défaut sur le crédit demandé :')
                    pie_sizes =[pred_proba, 1-pred_proba]
                    # Create a pieplot
                    plt.figure(figsize=(6,6))
                    plt.pie(pie_sizes, colors=[color, 'lightgrey'])
                    # add a circle at the center to transform it in a donut chart
                    my_circle=plt.Circle( (0,0), 0.7, color='white')
                    p=plt.gcf()
                    plt.text(0,0,'{:.1f}%'.format(pred_proba * 100.0),
                                horizontalalignment='center',
                                verticalalignment='center',
                                fontsize=30, fontweight='bold')
                    p.gca().add_artist(my_circle)
                    st.pyplot(p)

                with explain:
                    st.subheader('Facteurs de prédiction',
                                 anchor='predict-factors')
                    st_shap(shap_force_plot(details['shap_explainer'],
                                            details['shap_values'],
                                            details['Xp'],
                                            details['feature_names']))
                    st.write('**Principaux facteurs de prédiction '\
                             +'selon le modèle :**')
                    for col in ['EXT_SOURCE_1','EXT_SOURCE_2', 'EXT_SOURCE_3',
                                'AMT_GOODS_PRICE', 'AMT_CREDIT']:
                        if str(req[col]['value']) != "nan":
                            delta = gaps_req[col]['gap_med_pct']
                            pct = abs(delta * 100.0)
                            following = 'élevé que la médiane des demandes.'
                            if delta >= 0.0:
                                sign = 'plus'
                            else:
                                sign = 'moins'
                            s = '**{}**: {:.0f}% {} {}'.format(col, pct,
                                                            sign, following)
                            st.write(s)
                        else:
                            st.write('**{}**: valeur inconnue.'.format(col))

            main_infos = st.container()
            with main_infos:
                st.header('Informations sur le dossier N°{}'.format(req['id']))
                st.subheader('Information sur le demandeur')
                st.write(details['X'])
                age = floor(- req['DAYS_BIRTH']['value'] / 365)
                st.write("**ÂGE :** {}".format(age))
                st.write("**GENRE :** {}".format(req['CODE_GENDER']['value']))
                st.write("**PROFESSION :** {}".format(req['OCCUPATION_TYPE']\
                                                         ['value']))
                st.write("**REVENU ANNUEL :** {}".format(req['AMT_INCOME_TOTAL']\
                                                            ['value']))
                st.subheader('Information sur le prêt')
                st.subheader('Simuler d\'autres conditions d\'emprunt')
                new_simul = st.button('Simuler')