#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd

bur = pd.read_csv('../02_data/bureau.csv')
bur_bal = pd.read_csv('../02_data/bureau_balance.csv')
ccard_bal = pd.read_csv('../02_data/credit_card_balance.csv')
prev_app = pd.read_csv('../02_data/previous_application.csv')
pos_cash = pd.read_csv('../02_data/POS_CASH_balance.csv')
