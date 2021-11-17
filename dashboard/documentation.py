#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
sys.path.append('.')
sys.path.append('..')

import streamlit as st
from load_files import load_dataset
from individual import request_prediction, HEROKU_URL as API_URI

def display_descriptions():
    col_desc = load_dataset('HomeCredit_columns_description.csv', index_col=0,
                            encoding='latin9')
    desc_req = request_prediction(API_URI, '123456')
    #st.write(list(desc_req.keys())[:-10]) # debug
    for col in desc_req.keys():
        if type(desc_req[col]) == dict:
            if col[:7] != 'bureau_':
                desc = col_desc.loc[(col_desc.Table == 'application_{train|test}.csv')\
                                    & (col_desc.Row == col)].Description.values[0]
            else:
                gen_col = col[7:]
                metric = col.split('_')[-1]
                gen_col = gen_col.replace('_' + metric, '')
                #print(gen_col) # debug
                #print(metric) # debug
                desc = '{} of '.format(metric.capitalize())\
                     + col_desc.loc[(col_desc.Table == 'bureau.csv')\
                                    & (col_desc.Row == gen_col)].Description.values[0]
            st.write('**{}**: {}'.format(col, desc))

def display_main_doc():
    st.write(r'A prediction has to be over 67.5% of confidence that the application will have a'
             + r' default to consider the application as critically risky')


def display_documentation():
    st.subheader('Model risk threshold')
    display_main_doc()
    st.subheader('Variable descriptions')
    display_descriptions()