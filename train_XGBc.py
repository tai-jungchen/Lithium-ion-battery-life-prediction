"""
Filename: train_XGBc.py
Author: Alex
-----------------------------------------------------
This python program trains the battery data using XGBoost to classify
"""
from tqdm import tqdm
import math
import statistics as stat
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import confusion_matrix, classification_report
import xgboost as xgb
from sklearn.linear_model import LogisticRegression

THRESHOLD = 550
# THRESHOLD = 700

acc_list = []


def main():
    # load data #
    ########## load data ##########
    df = pd.read_pickle('processed/classification_voltage_only.pkl')
    # df = pd.read_pickle('processed/classification_full.pkl')
    ########## load data ##########
    data = df.to_numpy()
    print(f"data shape: {data.shape}")
    X = data[:, :-1]
    y = data[:, -1]
    y = np.where(y > THRESHOLD, 1, 0)  # label 1: long life; 0: short life

    ran_state = [5566, 2266, 22, 66, 521, 1126, 36, 819, 23, 1225]

    for i in range(len(ran_state)):
        # train test split #
        X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, stratify=y, test_size=0.33,
                                                                    random_state=ran_state[i])
        # hyper param search #
        print(f'\nSTART OF epoch {i}')
        lrs = [0.1, 1e-2, 1e-3, 1e-4]
        num_trees = [100, 200, 300, 400, 500]
        max_splits = [1, 2, 4]
        lambdas = [0]
        gammas = [0]
        cv_results = search_hyperparam_2(X_train_val, y_train_val, lrs, num_trees, max_splits, lambdas, gammas)

        # 5 fold cv #
        mean_accs = []
        se_accs = []
        for estimate in cv_results:
            lr, num_tree, max_split, lmb, gamma, accs, mean_acc, se_acc = estimate
            mean_accs.append(mean_acc)
            se_accs.append(se_acc)
        # plot cv acc #
        # plt.errorbar(np.linspace(1, len(cv_results), num=len(cv_results)), mean_accs, yerr=se_accs, fmt='o', ecolor='r', color='b')
        # plt.xlabel('hyper param sets')
        # plt.ylabel('Accuracy')
        # plt.title('5-fold cross validation on' + str(len(cv_results)) + 'hyper param sets')
        # plt.show()

        # Best model fit #
        index = mean_accs.index(max(mean_accs))
        best_lr, best_num_tree, best_max_split, best_lambda, best_gamma, _, _, _ = cv_results[index]
        reg = xgb.XGBClassifier(learning_rate=best_lr, n_estimators=best_num_tree, max_depth=best_max_split+1,
                               reg_lambda=best_lambda, gamma=best_gamma, use_label_encoder=False)
        reg.fit(X_train_val, y_train_val)

        ########## other models ##########
        model = LogisticRegression(solver='saga', penalty='elasticnet', l1_ratio=1)
        ########## other models ##########
        model.fit(X_train_val, y_train_val)

        # testing #
        print(f"\nbest_lr: {best_lr}, best_num_tree: {best_num_tree}, best_max_split: {best_max_split}, "
              f"best_lambda: {best_lambda}, best_gamma: {best_gamma}")
        print('<TESTING> - self')
        y_pred = reg.predict(X_test)
        acc = reg.score(X_test, y_test)
        cm = confusion_matrix(y_test, y_pred)
        cr = classification_report(y_test, y_pred)
        print(f"The testing accuracy:\n {acc}")
        print(f"The confusion matrix:\n {cm}")
        print(f"The classification report:\n {cr}")
        print(f'(END OF epoch {i})')
        print('-' * 40)
        # feature_importance(reg)
        acc_list.append(acc)

    print(f'mean accuracy: {stat.mean(acc_list)}')
    print(f'standard error: {stat.stdev(acc_list)/math.sqrt(10)}')
    print('\n END')


def search_hyperparam_2(X, y, lrs, num_trees, max_splits, lambdas, gammas):
    indexes = []
    for i in tqdm(range(len(lrs))):
        for j in range(len(num_trees)):
            for k in range(len(max_splits)):
                for l in range(len(lambdas)):
                    for m in range(len(gammas)):
                        model = xgb.XGBClassifier(learning_rate=lrs[i], n_estimators=num_trees[j],
                                                  max_depth=max_splits[k]+1, reg_lambda=lambdas[l], gamma=gammas[m],
                                                  use_label_encoder=False, verbosity=0)
                        accs = cross_val_score(model, X, y, cv=5, scoring='accuracy')
                        mean_acc = stat.mean(accs)
                        se_acc = stat.stdev(accs)/math.sqrt(5)
                        indexes.append((lrs[i], num_trees[j], max_splits[k], lambdas[l], gammas[m], accs, mean_acc, se_acc))
    return indexes


def feature_importance(model):
    """
    This function plots the feature importance of the model
    :param model: (obj) The model used for classifying
    """
    importance = model.feature_importances_
    # summarize feature importance
    for i, v in enumerate(importance):
        print('Feature: %0d, Score: %.5f' % (i, v))
    # plot feature importance
    plt.bar([x for x in range(len(importance))], importance)
    plt.title('Feature Importance Plot')
    plt.xlabel('Features')
    plt.ylabel('Feature importance')
    plt.show()


def mape_helper(y_true, y_pred):
    """
    This function helps calculating mape
    :param y_true: (np array) The true labels
    :param y_pred: (np array) The predicted labels
    :return:
    """
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


if __name__ == '__main__':
    main()
