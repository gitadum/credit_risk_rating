#! /usr/bin/env python3
# -*- coding: utf-8 -*-

#import os
#os.system("python api/app.py &")

import streamlit as st
from dashboard.individual import application_dashboard
from dashboard.app_base import display_applicantbase
from dashboard.raw_results import display_raw_results
from dashboard.documentation import display_documentation

st.sidebar.title("PretADepenser - KnowYourCustomer Dashboard")
tab = st.sidebar.radio("Select the desired view",
                       ("Individual", "Base", "API", "Documentation"))

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