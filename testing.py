"""
Filename: testing.py
Author: Alex
-----------------------------------------------------
This python program test the battery data using GBRT
"""
from tqdm import tqdm
import math
import statistics as stat
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

EPOCH = 4


def main():
    # load data #
    df = pd.read_pickle('Q_explored_data.pkl')
    data = df.to_numpy()
    print(f"data shape: {data.shape}")
    X = data[:, :data.shape[1]-1]
    # X = preprocessing.normalize(X)
    y = data[:, data.shape[1]-1]

    for epoch in range(EPOCH):
        # train test split #
        X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.33)
        # hyper param search #
        print(f'\nSTART OF epoch {epoch}')
        #################### hyper-param here ####################
        lr = 0.15
        num_tree = 700
        max_split = 2
        #################### hyper-param here ####################

        # Best model fit #
        reg = GradientBoostingRegressor(learning_rate=lr, n_estimators=num_tree, max_leaf_nodes=max_split+1)
        reg.fit(X_train_val, y_train_val)

        # testing #
        print('<TESTING>')
        y_pred = reg.predict(X_test)
        rmse = math.sqrt(mean_squared_error(y_test, y_pred))
        print("The root mean squared error (RMSE) on testing set: {:.4f}".format(rmse))
        mae = mean_absolute_error(y_test, y_pred)
        print("The mean absolute error (MAE) on testing set: {:.4f}".format(mae))
        mape = mape_helper(y_test, y_pred)
        print("The mean absolute percentage error (MAPE) on testing set: {:.4f}%".format(mape))
        r2 = r2_score(y_test, y_pred)
        print("The R^2 on testing set: {:.4f}".format(r2))
        r2_adj = 1-(1-r2)*(y.shape[0]-1)/(y.shape[0]-X.shape[1]-1)
        print("The Adjusted R^2 on testing set: {:.4f}".format(r2_adj))
        print(f'(END OF epoch {epoch})')
        print('-'*40)

        # importance of features #
        indices = np.argsort(reg.feature_importances_)
        # plot top 10 features as bar chart
        # plt.barh(np.arange(len(indices[:-11:-1])), reg.feature_importances_[indices][:-11:-1])
        plt.barh(np.arange(len(indices)), reg.feature_importances_[indices])
        plt.xlabel('Relative importance')
        plt.ylabel('feature indices')
        plt.title('Feature Importance')
        plt.yticks(range(1, 11))
        # plt.ylim(len(indices)-10, len(indices))
        plt.grid(True)
        plt.show()
        print(f"feature importance: {indices[::-1]}")

    print('\n END')


def mape_helper(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


if __name__ == '__main__':
    main()
