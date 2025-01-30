"""
Filename: test_XGB.py
Author: Alex
-----------------------------------------------------
This python program test the battery data using XGBoost to classify
"""
from tqdm import tqdm
import math
import statistics as stat
import xgboost as xgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import confusion_matrix, classification_report

EPOCH = 10
THRESHOLD = 550


def main():
    ########## load data ##########
    df = pd.read_pickle('explored_data.pkl')
    data = df.to_numpy()
    print(f"data shape: {data.shape}")
    X = data[:, :data.shape[1]-1]
    # X = preprocessing.normalize(X)
    y = data[:, data.shape[1]-1]
    y = np.where(y > THRESHOLD, 1, 0)  # label 1: long life; 0: short life
    ########## load data ##########

    for epoch in range(EPOCH):
        # train test split #
        X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.33)
        # hyper param search #
        print(f'\nSTART OF epoch {epoch}')
        #################### hyper-param here ####################
        lr = 0.2
        num_tree = 100
        max_split = 1
        lmb = 0
        gamma = 0
        #################### hyper-param here ####################

        # Best model fit #
        reg = xgb.XGBClassifier(learning_rate=lr, n_estimators=num_tree, max_depth=max_split + 1,
                               reg_lambda=lmb, gamma=gamma, use_label_encoder=False, verbosity=0)
        reg.fit(X_train_val, y_train_val)

        # testing #
        print('<TESTING>')
        y_pred = reg.predict(X_test)
        acc = reg.score(X_test, y_test)
        cm = confusion_matrix(y_test, y_pred)
        cr = classification_report(y_test, y_pred)
        print(f"The testing accuracy:\n {acc}")
        print(f"The confusion matrix:\n {cm}")
        print(f"The classification report:\n {cr}")
        print(f'(END OF epoch {epoch})')
        print('-'*40)

        # importance of features #
        # indices = np.argsort(reg.feature_importances_)
        # # plot top 10 features as bar chart
        # # plt.barh(np.arange(len(indices[:-11:-1])), reg.feature_importances_[indices][:-11:-1])
        # plt.barh(np.arange(len(indices)), reg.feature_importances_[indices])
        # plt.xlabel('Relative importance')
        # plt.ylabel('feature indices')
        # plt.title('Feature Importance')
        # plt.yticks(range(1, 11))
        # # plt.ylim(len(indices)-10, len(indices))
        # plt.grid(True)
        # plt.show()
        # print(f"feature importance: {indices[::-1]}")

    print('\n END')


def mape_helper(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


if __name__ == '__main__':
    main()
