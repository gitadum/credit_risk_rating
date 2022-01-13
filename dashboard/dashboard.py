#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import sys

sys.path.append('.')
sys.path.append('..')

import streamlit as st

from dashboard.individual import application_dashboard
from dashboard.app_base import display_applicantbase
from dashboard.raw_results import display_raw_results
from dashboard.documentation import display_documentation


st.sidebar.title("PretADepenser - KnowYourCustomer Dashboard")
tab = st.sidebar.radio(label="Select the desired view",
                       options=("Individual", "Base", "API", "Documentation"),
                       key='dashboardtabs')

if tab == "Individual":
    st.header("Individual Dashboard")
    application_dashboard()
elif tab == "Base":
    st.header("Application base viewer:")
    display_applicantbase()
elif tab == "API":
    st.header('API result displayer', anchor='raw-result')
    display_raw_results()
elif tab == "Documentation":
    st.header("Model Documentation")
    display_documentation()