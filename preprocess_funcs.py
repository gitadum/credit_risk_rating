#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

import warnings
warnings.simplefilter(action='ignore', category=UserWarning)


class DataframeFunctionTransformer():
    """FunctionTransformer object that works with pandas DataFrames
    -------
    source code : https://queirozf.com/entries/scikit-learn-pipelines-custom-pipelines-and-pandas-integration
    """
    def __init__(self, func):
        self.func = func

    def transform(self, input_df, **transform_params):
        return self.func(input_df)

    def fit(self, X, y=None, **fit_params):
        return self


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

        return [name + "__" + f for f in trans.get_feature_names()]
    
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


class SimpleImputerWithFeatureNames(SimpleImputer):
    """Thin wrapper around the SimpleImputer that provides get_feature_names()
    -------
    source code: https://github.com/benman1/OpenML-Speed-Dating/blob/master/openml_speed_dating_pipeline_steps/openml_speed_dating_pipeline_steps.py
    """
    def __init__(self, missing_values=np.nan, strategy="mean",
                 fill_value=None, verbose=0, copy=True):
        super(SimpleImputerWithFeatureNames, self).__init__(
            missing_values, strategy, fill_value, verbose,
            copy, add_indicator=True
        )

    def fit(self, X, y=None):
        super().fit(X, y)
        if isinstance(X, (pd.DataFrame, pd.Series)):
            self.features = list(X.columns)
        else:
            self.features = list(range(X.shape[1]))
        return self
  
    def transform(self, X):
        """Impute all missing values in X. Returns a DataFrame if given
        a DataFrame.
        
        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            The input data to complete.
        """
        X2 = super().transform(X)
        if isinstance(X, (pd.DataFrame, pd.Series)):
            return pd.DataFrame(
                data=X2, 
                columns=self.get_feature_names()
            )
        else:
            return X2
    
    def get_features_with_missing(self):
        return [self.features[f] for f in self.indicator_.features_]

    def get_feature_names(self):
        return self.features