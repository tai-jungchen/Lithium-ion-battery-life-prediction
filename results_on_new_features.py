"""
Author: (Alex) Tai-Jung Chen

Rerun the models on the new engineered features.
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV
import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error, r2_score
from xgboost import XGBRegressor


def main(models):
    train_df = pd.read_csv('datasets/training.csv')
    prim_test_df = pd.read_csv('datasets/prim_test.csv')
    sec_test_df = pd.read_csv('datasets/sec_test.csv')

    # drop b2c1 since it is a outlier
    prim_test_df.drop(index=21, inplace=True)

    X_train, y_train = train_df.set_index('cell_id').iloc[:, :-1], train_df.set_index('cell_id').iloc[:, -1]
    X_test_prim, y_test_prim = prim_test_df.set_index('cell_id').iloc[:, :-1], prim_test_df.set_index('cell_id').iloc[:, -1]
    X_test_sec, y_test_sec = sec_test_df.set_index('cell_id').iloc[:, :-1], sec_test_df.set_index('cell_id').iloc[:, -1]

    output = pd.DataFrame(columns=['model', 'train_rmse', 'prim_test_rmse', 'sec_test_rmse'])
    output.iloc[:, 0] = models

    # model
    for model in models:
        idx = models.index(model)
        model.fit(X_train, y_train)

        y_pred_train = model.predict(X_train)
        y_pred_prim = model.predict(X_test_prim)
        y_pred_sec = model.predict(X_test_sec)

        # print('Training')
        rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
        output.iloc[idx, 1] = rmse
        # mae = mean_absolute_error(y_train, y_pred_train)
        # mape = mean_absolute_percentage_error(y_train, y_pred_train)
        # print(f"RMSE: {rmse:.4f}")
        # print(f"MAE: {mae:.4f}")
        # print(f"MAPE: {mape:.4%}")

        # print('\nPrimary Testing')
        rmse = np.sqrt(mean_squared_error(y_test_prim, y_pred_prim))
        output.iloc[idx, 2] = rmse
        # mae = mean_absolute_error(y_test_prim, y_pred_prim)
        # mape = mean_absolute_percentage_error(y_test_prim, y_pred_prim)
        # print(f"RMSE: {rmse:.4f}")
        # print(f"MAE: {mae:.4f}")
        # print(f"MAPE: {mape:.4%}")

        # print('\nSecondary Testing')
        rmse = np.sqrt(mean_squared_error(y_test_sec, y_pred_sec))
        output.iloc[idx, 3] = rmse
        # mae = mean_absolute_error(y_test_sec, y_pred_sec)
        # mape = mean_absolute_percentage_error(y_test_sec, y_pred_sec)
        # print(f"RMSE: {rmse:.4f}")
        # print(f"MAE: {mae:.4f}")
        # print(f"MAPE: {mape:.4%}")
    output.to_csv("result.csv")


if __name__ == "__main__":
    MODELS = [LinearRegression(), SVR(), DecisionTreeRegressor(), RandomForestRegressor(),
              GradientBoostingRegressor(), XGBRegressor(), MLPRegressor(max_iter=1000, hidden_layer_sizes=(50,))]
    main(MODELS)
