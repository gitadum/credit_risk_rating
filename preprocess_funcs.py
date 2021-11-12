#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from sklearn.pipeline import Pipeline
from load_files import load_dataset

import warnings
warnings.simplefilter(action='ignore', category=UserWarning)

def get_feature_names(column_transformer):
    """Get feature names from all transformers.
    Returns
    -------
    feature_names : list of strings
        Names of the features produced by transform.
    -------
    source code : https://johaupt.github.io/scikit-learn/tutorial/python/data%20processing/ml%20pipeline/model%20interpretation/columnTransformer_feature_names.html
    """
    # Remove the internal helper function
    #check_is_fitted(column_transformer)
    
    # Turn loopkup into function for better handling with pipeline later
    def get_names(trans):
        # >> Original get_feature_names() method
        if trans == 'drop' or (
                hasattr(column, '__len__') and not len(column)):
            return []
        if trans == 'passthrough':
            if hasattr(column_transformer, '_df_columns'):
                if ((not isinstance(column, slice))
                        and all(isinstance(col, str) for col in column)):
                    return column
                else:
                    return column_transformer._df_columns[column]
            else:
                indices = np.arange(column_transformer._n_features)
                return ['x%d' % i for i in indices[column]]
        if not hasattr(trans, 'get_feature_names'):
        # >>> Change: Return input column names if no method avaiable
            # Turn error into a warning
            warnings.warn("Transformer %s (type %s) does not "
                                 "provide get_feature_names. "
                                 "Will return input column names if available"
                                 % (str(name), type(trans).__name__))
            # For transformers without a get_features_names method,
            # use the input names to the column transformer
            if column is None:
                return []
            else:
                return [name + "__" + f for f in column]

        return [name + "__" + f for f in trans.get_feature_names_out()]
    
    ### Start of processing
    feature_names = []
    
    # Allow transformers to be pipelines. Pipeline steps are named differently, 
    # so preprocessing is needed
    if type(column_transformer) == Pipeline:
        l_transformers = [(name, trans, None, None) 
                          for step, name, trans in column_transformer._iter()]
    else:
        # For column transformers, follow the original method
        l_transformers = list(column_transformer._iter(fitted=True))
    
    
    for name, trans, column, _ in l_transformers: 
        if type(trans) == Pipeline:
            # Recursive call on pipeline
            _names = get_feature_names(trans)
            # if pipeline has no transformer that returns names
            if len(_names)==0:
                _names = [name + "__" + f for f in column]
            feature_names.extend(_names)
        else:
            feature_names.extend(get_names(trans))
    
    return feature_names


def add_secondary_table_features(df):
    df = df.copy()
    bur = load_dataset('bureau.csv')
    idx = 'SK_ID_CURR'
    days_before_curr_app_mean = np.abs(bur.groupby(idx).DAYS_CREDIT.mean())
    df['bureau_DAYS_CREDIT_mean'] = df.join(
        days_before_curr_app_mean, how='left').DAYS_CREDIT
    df['bureau_DAYS_CREDIT_mean'].fillna(- 1.0, inplace=True)
    days_before_curr_app_min = np.abs(bur.groupby(idx).DAYS_CREDIT.max())
    df['bureau_DAYS_CREDIT_min'] = df.join(
        days_before_curr_app_min, how='left').DAYS_CREDIT
    df['bureau_DAYS_CREDIT_min'].fillna(- 1.0, inplace=True)
    n_active_credits = bur[bur.CREDIT_ACTIVE == 'Active'].groupby(idx)\
                                                         .CREDIT_ACTIVE.count()
    df['bureau_CREDIT_ACTIVE_count'] = df.join(
        n_active_credits, how='left').CREDIT_ACTIVE
    df.bureau_CREDIT_ACTIVE_count.fillna(0, inplace=True)
    mean_days_cred_overdue = bur.groupby(idx).CREDIT_DAY_OVERDUE.mean()
    df['bureau_CREDIT_DAY_OVERDUE_mean'] = df.join(
        mean_days_cred_overdue, how='left').CREDIT_DAY_OVERDUE
    df['bureau_CREDIT_DAY_OVERDUE_mean'].fillna(0, inplace=True)
    max_enddate = bur.groupby(idx).DAYS_CREDIT_ENDDATE.max()
    df['bureau_DAYS_CREDIT_ENDDATE_max'] = df.join(
        max_enddate, how='left').DAYS_CREDIT_ENDDATE
    df.bureau_DAYS_CREDIT_ENDDATE_max.fillna(0, inplace=True)
    return df
