#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import streamlit as st
from api.app import app_db

#@st.cache(suppress_st_warning=True)
def display_customerbase():
    display_table = app_db.copy()
    for col in list(display_table.columns):
        display_table[col] = display_table[col].astype('category')
    st.dataframe(display_table)
