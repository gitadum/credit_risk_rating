#! /usr/bin/env python3
# -*- coding: utf-8 -*-

from sklearn.metrics import confusion_matrix, classification_report
from timer import timer

# Fonction d'évaluation des modèles
@timer
def model_eval(model, X_test, y_test):
    y_pred = model.predict(X_test)
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))