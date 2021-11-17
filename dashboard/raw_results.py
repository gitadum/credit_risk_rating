#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import streamlit as st
from individual import request_prediction, HEROKU_URL as API_URI
from api.app import gap_with_trends

def display_raw_results():
    ex_id_input = st.text_input(label='Enter an application id number')
    ex_predict_btn = st.button('View')
    if ex_predict_btn:
        ex_req = request_prediction(API_URI, ex_id_input)
        ex_gaps_req = gap_with_trends(ex_req)
        st.write(ex_req)
        st.write(ex_gaps_req)