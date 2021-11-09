#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
sys.path.append('..')

import streamlit as st
from customer import customer_dashboard
from customer_base import display_customerbase

st.sidebar.title("PretADepenser - KYC Dashboard")
tab = st.sidebar.radio("Sélectionnez l'onglet souhaité",
                       ("KYC", "Base client", "Autre"))

if tab == "KYC":
    st.header("Tableau de bord individuel")
    customer_dashboard()
elif tab == "Base client":
    st.header("Base client :")
    display_customerbase()
elif tab == "Autre":
    st.header('Modèle', anchor='model')
    st.write("Autres options (prochainement)")
