#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import joblib

def load_dataset(filename, data_path='02_data', *args, **kwargs):
    try:
        file = pd.read_csv(data_path + '/' + filename, *args, **kwargs)
    except FileNotFoundError:
        try:
            file = pd.read_csv('../' + data_path + '/' + filename,
                               *args, **kwargs)
        except FileNotFoundError:
            raise FileNotFoundError('{} not found in {} dir.'.format(filename,
                                                                     data_path))
    return file


def load_model(filename, data_path='model', *args, **kwargs):
    try:
        model = joblib.load(data_path + '/' + filename, *args, **kwargs)
    except FileNotFoundError:
        try:
            model = joblib.load('../' + data_path + '/' + filename,
                                *args, **kwargs)
        except FileNotFoundError:
            raise FileNotFoundError('{} not found in {} dir.'.format(filename,
                                                                     data_path))
    return model