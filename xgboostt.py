import re
from datetime import datetime, timedelta
import xgboost as xgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor  # tf4
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error

month2days = {"1": 31, "2": 28, "3": 31, "4": 30, "5": 31, "6": 30, "7": 31, "8": 31, "9": 30, "10": 31, "11": 30,
              "12": 31}

"""
reference
- https://www.cnblogs.com/wj-1314/p/9402324.html
- https://zhuanlan.zhihu.com/p/28672955
- https://blog.csdn.net/han_xiaoyang/article/details/52665396
- https://blog.csdn.net/sinat_35512245/article/details/79700029
"""

train_csv_path = "train.csv"
test_csv_path = 'test.csv'


def create_data_simple(train_prob=0.8):
    """
    X = [year, day, month, hour, weekday]
    y = [speed]
    """
    # ####################read train data set######################
    file = pd.read_csv(train_csv_path)
    df = pd.DataFrame(file)
    X = []
    y = []

    for row in df.iterrows():
        # "date and time" in train set & create later time
        cur_date_and_time = datetime.strptime(row[1]["date"], "%d/%m/%Y %H:%M")  # string -> datetime
        speed = row[1]["speed"]
        xi = np.zeros(shape=(5,))

        # "date and time" in test set
        xi[0] = int(cur_date_and_time.year)
        xi[1] = float(cur_date_and_time.day) / month2days[str(cur_date_and_time.month)]
        xi[2] = float(cur_date_and_time.month)
        xi[3] = float(cur_date_and_time.hour)
        xi[4] = cur_date_and_time.weekday()
        y.append(speed)
        X.append(xi)

    # ###########################read test data set#########################
    file = pd.read_csv(test_csv_path)
    df = pd.DataFrame(file)
    test_set = []
    for row in df.iterrows():
        # "date and time" in test set & create later time
        cur_date_and_time = datetime.strptime(row[1]["date"], "%d/%m/%Y %H:%M")  # string -> datetime
        xi = np.zeros(shape=(5,))

        # "date and time" in test set
        xi[0] = int(cur_date_and_time.year)
        xi[1] = float(cur_date_and_time.day) / month2days[str(cur_date_and_time.month)]
        xi[2] = float(cur_date_and_time.month)
        xi[3] = float(cur_date_and_time.hour)
        xi[4] = cur_date_and_time.weekday()
        test_set.append(xi)

    # to array
    X = np.array(X, dtype=float)
    y = np.array(y, dtype=float)
    test_set = np.array(test_set, dtype=float)

    # shuffle
    total = X.shape[0]
    index = [i for i in range(total)]
    np.random.shuffle(index)
    np.random.shuffle(index)
    np.random.shuffle(index)
    X = X[index]
    y = y[index]

    X_train = X[0: int(total * train_prob)]
    y_train = y[0: int(total * train_prob)]
    X_test = X[int(total * train_prob):]
    y_test = y[int(total * train_prob):]

    return X_train, y_train, X_test, y_test, test_set


def create_data_add_weektime_ma_bachforth_nonmap(train_prob=0.8):
    """
    consider hours that before current time and after current time
    X = [year, day, month, hour, weekday] * before * 2
    y = [speed]
    """
    # ####################read train data set######################
    file = pd.read_csv('train.csv')
    df = pd.DataFrame(file)
    X = []
    y = []
    peroid = 7

    for row in df.iterrows():
        # "date and time" in train set & create later time
        cur_date_and_time = datetime.strptime(row[1]["date"], "%d/%m/%Y %H:%M")  # string -> datetime
        speed = row[1]["speed"]
        xi = np.zeros(shape=(peroid * 2 * 5,))  # features of one timestamp * 3

        # "date and time" in test set
        xi[0] = int(cur_date_and_time.year)
        xi[1] = float(cur_date_and_time.day)
        xi[2] = float(cur_date_and_time.month)
        xi[3] = float(cur_date_and_time.hour)
        xi[4] = cur_date_and_time.weekday()
        y.append(speed)

        j = 1
        while j < peroid:
            cur_date_and_time = cur_date_and_time - timedelta(hours=1)
            xi[j * 5 + 0] = int(cur_date_and_time.year)
            xi[j * 5 + 1] = float(cur_date_and_time.day)
            xi[j * 5 + 2] = float(cur_date_and_time.month)
            xi[j * 5 + 3] = float(cur_date_and_time.hour)
            xi[j * 5 + 4] = cur_date_and_time.weekday()
            j += 1

        while j < peroid * 2:
            cur_date_and_time = cur_date_and_time + timedelta(hours=1)
            xi[j * 5 + 0] = int(cur_date_and_time.year)
            xi[j * 5 + 1] = float(cur_date_and_time.day)
            xi[j * 5 + 2] = float(cur_date_and_time.month)
            xi[j * 5 + 3] = float(cur_date_and_time.hour)
            xi[j * 5 + 4] = cur_date_and_time.weekday()
            j += 1
        X.append(xi)

    # ###########################read test data set#########################
    file = pd.read_csv('test.csv')
    df = pd.DataFrame(file)
    test_set = []
    for row in df.iterrows():
        # "date and time" in test set & create later time
        cur_date_and_time = datetime.strptime(row[1]["date"], "%d/%m/%Y %H:%M")  # string -> datetime
        xi = np.zeros(shape=(peroid * 2 * 5,))  # features of one timestamp * 3

        # "date and time" in test set
        xi[0] = int(cur_date_and_time.year)
        xi[1] = float(cur_date_and_time.day)
        xi[2] = float(cur_date_and_time.month)
        xi[3] = float(cur_date_and_time.hour)
        xi[4] = cur_date_and_time.weekday()

        j = 1
        while j < peroid:
            cur_date_and_time = cur_date_and_time - timedelta(hours=1)
            xi[j * 5 + 0] = int(cur_date_and_time.year)
            xi[j * 5 + 1] = float(cur_date_and_time.day)
            xi[j * 5 + 2] = float(cur_date_and_time.month)
            xi[j * 5 + 3] = float(cur_date_and_time.hour)
            xi[j * 5 + 4] = cur_date_and_time.weekday()
            j += 1

        while j < peroid * 2:
            cur_date_and_time = cur_date_and_time + timedelta(hours=1)
            xi[j * 5 + 0] = int(cur_date_and_time.year)
            xi[j * 5 + 1] = float(cur_date_and_time.day)
            xi[j * 5 + 2] = float(cur_date_and_time.month)
            xi[j * 5 + 3] = float(cur_date_and_time.hour)
            xi[j * 5 + 4] = cur_date_and_time.weekday()
            j += 1
        test_set.append(xi)

    # to array
    X = np.array(X, dtype=float)
    y = np.array(y, dtype=float)
    test_set = np.array(test_set, dtype=float)

    # shuffle
    total = X.shape[0]
    index = [i for i in range(total)]
    np.random.shuffle(index)
    np.random.shuffle(index)
    np.random.shuffle(index)
    X = X[index]
    y = y[index]

    X_train = X[0: int(total * train_prob)]
    y_train = y[0: int(total * train_prob)]
    X_test = X[int(total * train_prob):]
    y_test = y[int(total * train_prob):]

    return X_train, y_train, X_test, y_test, test_set


def create_data_add_weektime_ma_bachforth(train_prob=0.8):
    """
    consider hours that before current time and after current time
    add some other features (boolean)
    X = [year, day, month, hour, weekday, morningpeek, eveningpeek] * before * 2
    y = [speed]
    """
    # ####################read train data set######################
    file = pd.read_csv('train.csv')
    df = pd.DataFrame(file)
    X = []
    y = []
    peroid = 7

    for row in df.iterrows():
        # "date and time" in train set & create later time
        cur_date_and_time = datetime.strptime(row[1]["date"], "%d/%m/%Y %H:%M")  # string -> datetime
        speed = row[1]["speed"]
        xi = np.zeros(shape=(peroid * 2 * 7,))  # features of one timestamp * 3

        # "date and time" in test set
        xi[0] = int(cur_date_and_time.year == 2017)
        xi[1] = float(cur_date_and_time.day) / month2days[str(cur_date_and_time.month)]
        xi[2] = float(cur_date_and_time.month) / 12
        xi[3] = float(cur_date_and_time.hour) / 24
        xi[4] = cur_date_and_time.weekday() / 7
        if 8 <= float(cur_date_and_time.hour) <= 10:  # 是否早高峰
            xi[5] = 1
        else:
            xi[5] = 0
        if 17 <= float(cur_date_and_time.hour) <= 19:
            xi[6] = 1
        else:
            xi[6] = 0
        y.append(speed)

        j = 1
        while j < peroid:
            cur_date_and_time = cur_date_and_time - timedelta(hours=1)
            xi[j * 7 + 0] = int(cur_date_and_time.year == 2017)
            xi[j * 7 + 1] = float(cur_date_and_time.day) / month2days[str(cur_date_and_time.month)]
            xi[j * 7 + 2] = float(cur_date_and_time.month) / 12
            xi[j * 7 + 3] = float(cur_date_and_time.hour) / 24
            xi[j * 7 + 4] = cur_date_and_time.weekday() / 7
            if 8 <= float(cur_date_and_time.hour) <= 10:  # 是否早高峰
                xi[j * 7 + 5] = 1
            else:
                xi[j * 7 + 5] = 0
            if 17 <= float(cur_date_and_time.hour) <= 19:
                xi[j * 7 + 6] = 1
            else:
                xi[j * 7 + 6] = 0
            j += 1

        while j < peroid * 2:
            cur_date_and_time = cur_date_and_time + timedelta(hours=1)
            xi[j * 7 + 0] = int(cur_date_and_time.year == 2017)
            xi[j * 7 + 1] = float(cur_date_and_time.day) / month2days[str(cur_date_and_time.month)]
            xi[j * 7 + 2] = float(cur_date_and_time.month) / 12
            xi[j * 7 + 3] = float(cur_date_and_time.hour) / 24
            xi[j * 7 + 4] = cur_date_and_time.weekday() / 7
            if 8 <= float(cur_date_and_time.hour) <= 10:  # 是否早高峰
                xi[j * 7 + 5] = 1
            else:
                xi[j * 7 + 5] = 0
            if 17 <= float(cur_date_and_time.hour) <= 19:
                xi[j * 7 + 6] = 1
            else:
                xi[j * 7 + 6] = 0
            j += 1
        X.append(xi)

    # ###########################read test data set#########################
    file = pd.read_csv('test.csv')
    df = pd.DataFrame(file)
    test_set = []
    for row in df.iterrows():
        # "date and time" in test set & create later time
        cur_date_and_time = datetime.strptime(row[1]["date"], "%d/%m/%Y %H:%M")  # string -> datetime
        xi = np.zeros(shape=(peroid * 2 * 7,))  # features of one timestamp * 3

        # "date and time" in test set
        xi[0] = int(cur_date_and_time.year == 2017)
        xi[1] = float(cur_date_and_time.day) / month2days[str(cur_date_and_time.month)]
        xi[2] = float(cur_date_and_time.month) / 12
        xi[3] = float(cur_date_and_time.hour) / 24
        xi[4] = cur_date_and_time.weekday() / 7
        if 8 <= float(cur_date_and_time.hour) <= 10:  # 是否早高峰
            xi[5] = 1
        else:
            xi[5] = 0
        if 17 <= float(cur_date_and_time.hour) <= 19:
            xi[6] = 1
        else:
            xi[6] = 0

        j = 1
        while j < peroid:
            cur_date_and_time = cur_date_and_time - timedelta(hours=1)
            xi[j * 7 + 0] = int(cur_date_and_time.year == 2017)
            xi[j * 7 + 1] = float(cur_date_and_time.day) / month2days[str(cur_date_and_time.month)]
            xi[j * 7 + 2] = float(cur_date_and_time.month) / 12
            xi[j * 7 + 3] = float(cur_date_and_time.hour) / 24
            xi[j * 7 + 4] = cur_date_and_time.weekday() / 7
            if 8 <= float(cur_date_and_time.hour) <= 10:  # 是否早高峰
                xi[j * 7 + 5] = 1
            else:
                xi[j * 7 + 5] = 0
            if 17 <= float(cur_date_and_time.hour) <= 19:
                xi[j * 7 + 6] = 1
            else:
                xi[j * 7 + 6] = 0
            j += 1

        while j < peroid * 2:
            cur_date_and_time = cur_date_and_time + timedelta(hours=1)
            xi[j * 7 + 0] = int(cur_date_and_time.year == 2017)
            xi[j * 7 + 1] = float(cur_date_and_time.day) / month2days[str(cur_date_and_time.month)]
            xi[j * 7 + 2] = float(cur_date_and_time.month) / 12
            xi[j * 7 + 3] = float(cur_date_and_time.hour) / 24
            xi[j * 7 + 4] = cur_date_and_time.weekday() / 7
            if 8 <= float(cur_date_and_time.hour) <= 10:  # 是否早高峰
                xi[j * 7 + 5] = 1
            else:
                xi[j * 7 + 5] = 0
            if 17 <= float(cur_date_and_time.hour) <= 19:
                xi[j * 7 + 6] = 1
            else:
                xi[j * 7 + 6] = 0
            j += 1
        test_set.append(xi)

    # to array
    X = np.array(X, dtype=float)
    y = np.array(y, dtype=float)
    test_set = np.array(test_set, dtype=float)

    # shuffle
    total = X.shape[0]
    index = [i for i in range(total)]
    np.random.shuffle(index)
    np.random.shuffle(index)
    np.random.shuffle(index)
    np.random.shuffle(index)
    np.random.shuffle(index)
    X = X[index]
    y = y[index]

    X_train = X[0: int(total * train_prob)]
    y_train = y[0: int(total * train_prob)]
    X_test = X[int(total * train_prob):]
    y_test = y[int(total * train_prob):]

    return X_train, y_train, X_test, y_test, test_set


def feature_importances(model, feature_desctiption):
    print(model.feature_importances_)  # use inbuilt class feature_importances
    # plot graph of feature importances for better visualization
    feat_importances = pd.Series(model.feature_importances_, index=feature_desctiption)
    feat_importances.nlargest(5).plot(kind='barh')
    plt.show()


def final_predict(model, model_name, test_set):
    y_pred = model.predict(test_set)
    file = pd.read_csv(r'test.csv') 
    file['speed'] = y_pred 
    file.to_csv(r"test_submit_{model}.csv".format(model=model_name), mode='a')


def print_best_score(gsearch, param_test):
    print("Best score: %0.3f" % gsearch.best_score_)
    print("Best parameters set:")
    best_parameters = gsearch.best_estimator_.get_params()
    for param_name in sorted(param_test.keys()):
        print("\t%s: %r" % (param_name, best_parameters[param_name]))


def main():
    X_train, y_train, X_test, y_test, test_set = create_data_simple()
    model = xgb.XGBRegressor(max_depth=6, learning_rate=0.07, n_estimators=650, min_child_weight=5,
                             subsample=0.8, colsample_bytree=0.9,
                             gamma=0.5, reg_alpha=2.7, reg_lambda=3, objective='reg:linear')
    # model = xgb.XGBRegressor(max_depth=5, min_child_weight=1, gamma=0, 
    #                          subsample=0.8, colsample_bytree=0.8,
    #                          learning_rate=0.1, eta = 0.1, n_estimators=1000, 
    #                          objective='reg:linear', 
    #                          booster='gbtree', n_jobs=4, 
    #                          colsample_bylevel=1, colsample_bynode=1, 
    #                          reg_alpha=0, reg_lambda=1)

    # cv grid search
    param_test0 = {
        # 'n_estimators': [100, 130, 200, 500, 1000]
        'n_estimators': [550, 575, 600, 650, 675]
    }
    param_test1 = {
        'max_depth': range(3, 10, 1),
        'min_child_weight': range(1, 6, 1)
    } 
    param_test2 = {
        'max_depth': [5, 6, 7],
        'min_child_weight': [4, 5, 6]
    } 
    param_test3 = {
        # 'gamma': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
        'gamma': np.arange(0.45, 0.55, 0.01)
    } 
    param_test4 = {
        'subsample': [0.5, 0.8], 
        'colsample_bytree': [0.5, 0.9]
    }
    param_test5 = {
        # 'reg_alpha': [0.05, 0.1, 1, 2, 3], 'reg_lambda': [0.05, 0.1, 1, 2, 3]
        'reg_alpha': [2.7, 3, 3.3], 'reg_lambda': [2.7, 3, 3.3]
        # 'reg_alpha': [1e-5, 1e-2, 0.1, 1, 100]
        # 'reg_alpha': np.arange(0.05, 0.15, 0.01)
    }
    param_test6 = {
        'learning_rate': [0.01, 0.05, 0.07, 0.1, 0.2]
    }
    # gsearch = GridSearchCV(model, param_grid=param_test6, scoring='neg_mean_squared_error', cv=5)
    # gsearch.fit(X_train, y_train)
    model.fit(X_train, y_train)

    # evaluate model
    # print_best_score(gsearch, param_test6)
    # prec = model.score(X_test, y_test)
    prec = mean_squared_error(model.predict(X_test), y_test)
    print(prec)

    # final_predict(model, "xgboost", test_set)


if __name__ == '__main__':
    main()
