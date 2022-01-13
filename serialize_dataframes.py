#! /usr/bin/env python3
# -*- coding: utf-8 -*-

# In[1]:
import os
import pandas as pd

# In[2]:
os.mkdir('data_files')

# In[3]:
app_test = pd.read_csv('02_data/application_test.csv').loc[::10]
app_test.to_pickle('data_files/application_test.bz2')

# In[4]:
app_train = pd.read_csv('02_data/application_train.csv').loc[::10]
app_train.to_pickle('data_files/application_train.bz2')

# In[5]:
bureau = pd.read_csv('02_data/bureau.csv')
bureau.to_pickle('data_files/bureau.bz2')

# In[6]:
col_desc = pd.read_csv('02_data/HomeCredit_columns_description.csv',
                       index_col=0,
                       encoding='latin9')
col_desc.to_pickle('data_files/HomeCredit_columns_description.bz2')
