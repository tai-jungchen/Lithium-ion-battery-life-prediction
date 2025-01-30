"""
Filename: train_XGB.py
Author: Alex
-----------------------------------------------------
This python program trains the battery data using XGBoost
"""
from tqdm import tqdm
import math
import statistics as stat
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb

EPOCH = 20


def main():
    # load data #
    df = pd.read_pickle('processed/explored_data.pkl')
    data = df.to_numpy()
    print(f"data shape: {data.shape}")
    X = data[:, :data.shape[1]-1]
    y = data[:, data.shape[1]-1]

    for epoch in range(EPOCH):
        # train test split #
        X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.33)
        # hyper param search #
        print(f'\nSTART OF epoch {epoch}')
        lrs = [0.005, 0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.5]
        num_trees = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
        max_splits = [1, 2, 4, 8, 16, 32, 64]
        lambdas = np.linspace(1, 100, num=10)
        gammas = np.linspace(1, 100, num=10)
        cv_results = search_hyperparam_2(X_train_val, y_train_val, lrs, num_trees, max_splits, lambdas, gammas)

        # 5 fold cv #
        mean_cres = []
        se_cres = []
        for estimate in cv_results:
            lr, num_tree, max_split, lmb, gamma, cres, mean_cre, se_cre = estimate
            mean_cres.append(mean_cre)
            se_cres.append(se_cre)
        # plot cv rmse #
        plt.errorbar(np.linspace(1, len(cv_results), num=len(cv_results)), mean_cres, yerr=se_cres, fmt='o', ecolor='r', color='b')
        plt.xlabel('hyper param sets')
        plt.ylabel('RMSE')
        plt.title('5-fold cross validation on ' + str(len(cv_results)) + ' hyper param sets')
        plt.show()

        # Best model fit #
        index = mean_cres.index(min(mean_cres))
        best_lr, best_num_tree, best_max_split, best_lambda, best_gamma, _, _, _ = cv_results[index]
        reg = xgb.XGBRegressor(learning_rate=best_lr, n_estimators=best_num_tree, max_depth=best_max_split+1,
                               reg_lambda=best_lambda, gamma=best_gamma)
        reg.fit(X_train_val, y_train_val)

        # testing #
        print(f"\nbest_lr: {best_lr}, best_num_tree: {best_num_tree}, best_max_split: {best_max_split}, "
              f"best_lambda: {best_lambda}, best_gamma: {best_gamma}")
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
        # indices = np.argsort(reg.feature_importances_)
        # plot top 10 features as bar chart
        # plt.barh(np.arange(len(indices[:-11:-1])), reg.feature_importances_[indices][:-11:-1])
        # plt.barh(np.arange(len(indices)), reg.feature_importances_[indices])
        # plt.xlabel('Relative importance')
        # plt.ylabel('feature indices')
        # plt.title('Feature Importance')
        # plt.yticks(range(1, 11))
        # plt.ylim(len(indices)-10, len(indices))
        # plt.grid(True)
        # plt.show()

    print('\n END')


def search_hyperparam_2(X, y, lrs, num_trees, max_splits, lambdas, gammas):
    indexes = []
    for i in tqdm(range(len(lrs))):
        for j in range(len(num_trees)):
            for k in range(len(max_splits)):
                for l in range(len(lambdas)):
                    for m in range(len(gammas)):
                        model = xgb.XGBRegressor(learning_rate=lrs[i], n_estimators=num_trees[j], max_depth=max_splits[k]+1,
                                                 reg_lambda= lambdas[l], gamma=gammas[m])
                        cres = cross_val_score(model, X, y, cv=5, scoring='neg_root_mean_squared_error')*(-1)
                        mean_cre = stat.mean(cres)
                        se_cre = stat.stdev(cres)/math.sqrt(5)
                        indexes.append((lrs[i], num_trees[j], max_splits[k], lambdas[l], gammas[m], cres, mean_cre, se_cre))
    return indexes


def mape_helper(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


if __name__ == '__main__':
    main()
