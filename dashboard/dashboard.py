#! /usr/bin/env python3
# -*- coding: utf-8 -*-

#import os
#os.system("python api/app.py &")

import streamlit as st

from individual import application_dashboard
from app_base import display_applicantbase
from raw_results import display_raw_results
from documentation import display_documentation


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