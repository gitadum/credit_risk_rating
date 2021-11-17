#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import streamlit as st
from customer import application_dashboard
from customer_base import display_applicantbase
from raw_results import display_raw_results

st.sidebar.title("PretADepenser - KnowYourCustomer Dashboard")
tab = st.sidebar.radio("Select the desired view",
                       ("Individual", "Base", "API"))

if tab == "Individual":
    st.header("Individual Dashboard")
    application_dashboard()
elif tab == "Base":
    st.header("Application base viewer:")
    display_applicantbase()
elif tab == "API":
    st.header('API result displayer', anchor='raw-result')
    display_raw_results()
