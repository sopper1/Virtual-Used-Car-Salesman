'''
Name: Shawn Oppermann
File Desc: a list of functions for use with Final Project Report.ipynb
'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
import pytz

import category_encoders as ce

from sklearn.model_selection import KFold
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn import metrics

def plot_categorical_means(car_data, car_data_cols):
    
    def qual_count(ax, df, group_name, label_name):
        x = df[group_name].unique()
        y = df.groupby(df[group_name])[label_name].count()
        ax.bar(x, y)
        ax.set_title(group_name + " counts")
    
    def qual_mean(ax, df, group_name, label_name):
        x = df[group_name].unique()
        y = df.groupby(df[group_name])[label_name].mean()
        ax.bar(x, y)
        ax.set_title(group_name + " mean prices")

    qualitative_cols = car_data_cols.copy()
    qualitative_cols.remove('price')
    qualitative_cols.remove('odometer')

    fig, ax = plt.subplots(len(qualitative_cols), figsize = (20, 20))
    fig.tight_layout()

    for i, col in enumerate(qualitative_cols):
        qual_mean(ax[i], car_data, col, 'price')

    plt.show()

def train(X, y):
    
    ridge_models = [None] * 15
    lasso_models = [None] * 15

    ridge_scores = [None] * 15
    lasso_scores = [None] * 15

    alpha_grid = [.25, .5, 1, 2, 4] * 3
    degree = [1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3]
    
    kf = KFold(n_splits = 15)

    from warnings import simplefilter
    from sklearn.exceptions import ConvergenceWarning
    # ignore all future and convergence warnings
    simplefilter(action='ignore', category=FutureWarning)
    simplefilter(action='ignore', category=ConvergenceWarning)

    i = 0
    for train_i, test_i in kf.split(X, y):

        X_train, X_test = X.iloc[train_i], X.iloc[test_i]
        y_train, y_test = y.iloc[train_i], y.iloc[test_i]

        encoder = ce.TargetEncoder(cols=['manufacturer', 'state', 'condition', 'cylinders', 'fuel', 'title_status', 'transmission', 'drive', 'size', 'type', 'paint_color'])
        poly = PolynomialFeatures(degree = degree[i])
        scaler = StandardScaler()

        encoder.fit(X_train, y_train)
        X_train = encoder.transform(X_train)
        X_test = encoder.transform(X_test)

        poly.fit(X_train, y_train)
        X_train = poly.transform(X_train)
        X_test = poly.transform(X_test)

        scaler.fit(X_train)
        X_train = scaler.transform(X_train)
        X_test = scaler.transform(X_test)

        ridge_models[i] = Ridge(alpha = alpha_grid[i])
        lasso_models[i] = Lasso(alpha = alpha_grid[i])

        ridge_models[i].fit(X_train, y_train)
        lasso_models[i].fit(X_train, y_train)

        ridge_scores[i] = ridge_models[i].score(X_test, y_test)
        lasso_scores[i] = lasso_models[i].score(X_test, y_test)

        print("alpha =", alpha_grid[i], ", p-degree =", degree[i])
        print("   ", "Ridge Score:", ridge_scores[i])
        print("   ", "Lasso Score:", lasso_scores[i])

        i += 1
        
    best_ridge_model = ridge_models[ridge_scores.index(max(ridge_scores))]
    best_lasso_model = lasso_models[lasso_scores.index(max(lasso_scores))]
    
    return best_ridge_model, best_lasso_model, X_test, y_test

def test_accuracy(ridge_model, lasso_model, X_test, y_test):
    
    ridge_predictions = ridge_model.predict(X_test)
    lasso_predictions = lasso_model.predict(X_test)
    ridge_acc = metrics.mean_absolute_error(y_test, ridge_predictions)
    lasso_acc = metrics.mean_absolute_error(y_test, lasso_predictions)

    print("Ridge Mean Absolute Error:", ridge_acc)
    print("Lasso Mean Absolute Error:", lasso_acc)