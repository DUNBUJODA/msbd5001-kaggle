import numpy as np
import re
from sklearn.ensemble import RandomForestRegressor  # tf4
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

month2days = {"1": 31, "2": 28, "3": 31, "4": 30, "5": 31, "6": 30, "7": 31, "8": 31, "9": 30, "10": 31, "11": 30,
              "12": 31}


def create_data_add_weektime_ma(train_prob=0.8):
    """
    认为前几天的时间也可以纳入特征，一行向量表示
    :param train_prob:
    :return:
    """
    # ####################read train data set######################
    file = pd.read_csv('train.csv')
    df = pd.DataFrame(file)
    X = []
    y = []
    before = 14

    for row in df.iterrows():
        # "date and time" in train set & create later time
        cur_date_and_time = datetime.strptime(row[1]["date"], "%d/%m/%Y %H:%M")  # string -> datetime
        speed = row[1]["speed"]
        xi = np.zeros(shape=(5 * before,))  # features of one timestamp * 3

        # "date and time" in test set
        xi[0] = int(cur_date_and_time.year == 2017)
        xi[1] = float(cur_date_and_time.day) / month2days[str(cur_date_and_time.month)]
        xi[2] = float(cur_date_and_time.month) / 12
        xi[3] = float(cur_date_and_time.hour) / 24
        xi[4] = cur_date_and_time.weekday() / 7
        y.append(speed)

        j = 1
        while j < before:
            cur_date_and_time = cur_date_and_time - timedelta(hours=1)
            xi[j * 5 + 0] = int(cur_date_and_time.year == 2017)
            xi[j * 5 + 1] = float(cur_date_and_time.day) / month2days[str(cur_date_and_time.month)]
            xi[j * 5 + 2] = float(cur_date_and_time.month) / 12
            xi[j * 5 + 3] = float(cur_date_and_time.hour) / 24
            xi[j * 5 + 4] = cur_date_and_time.weekday() / 7
            j += 1
        X.append(xi)

    # ###########################read test data set#########################
    file = pd.read_csv('test.csv')
    df = pd.DataFrame(file)
    test_set = []
    for row in df.iterrows():
        # "date and time" in test set & create later time
        cur_date_and_time = datetime.strptime(row[1]["date"], "%d/%m/%Y %H:%M")  # string -> datetime
        xi = np.zeros(shape=(5 * before,))  # features of one timestamp * 3

        # "date and time" in test set
        xi[0] = int(cur_date_and_time.year == 2017)
        xi[1] = float(cur_date_and_time.day) / month2days[str(cur_date_and_time.month)]
        xi[2] = float(cur_date_and_time.month) / 12
        xi[3] = float(cur_date_and_time.hour) / 24
        xi[4] = cur_date_and_time.weekday() / 7

        j = 1
        while j < before:
            cur_date_and_time = cur_date_and_time - timedelta(hours=1)
            xi[j * 5 + 0] = int(cur_date_and_time.year == 2017)
            xi[j * 5 + 1] = float(cur_date_and_time.day) / month2days[str(cur_date_and_time.month)]
            xi[j * 5 + 2] = float(cur_date_and_time.month) / 12
            xi[j * 5 + 3] = float(cur_date_and_time.hour) / 24
            xi[j * 5 + 4] = cur_date_and_time.weekday() / 7
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
    np.random.shuffle(index)

    X_train = X[0: int(total * train_prob)]
    y_train = y[0: int(total * train_prob)]
    X_test = X[int(total * train_prob):]
    y_test = y[int(total * train_prob):]

    return X_train, y_train, X_test, y_test, test_set


def create_data_add_weektime_ma_matrix(train_prob=0.8):
    """
    前一段时间也可以纳入特征，以矩阵表示(sklearn不能用矩阵，转到cnn.py看)
    :param train_prob:
    :return:
    """
    # ####################read train data set######################
    file = pd.read_csv('train.csv')
    df = pd.DataFrame(file)
    X = []
    y = []
    before = 14

    for row in df.iterrows():
        # "date and time" in train set & create later time
        cur_date_and_time = datetime.strptime(row[1]["date"], "%d/%m/%Y %H:%M")  # string -> datetime
        speed = row[1]["speed"]
        xi = np.zeros(shape=(before, 5))  # features of one timestamp * 3

        # "date and time" in test set
        xi[0, 0] = int(cur_date_and_time.year == 2017)
        xi[0, 1] = float(cur_date_and_time.day) / month2days[str(cur_date_and_time.month)]
        xi[0, 2] = float(cur_date_and_time.month) / 12
        xi[0, 3] = float(cur_date_and_time.hour) / 24
        xi[0, 4] = cur_date_and_time.weekday() / 7
        y.append(speed)

        j = 1
        while j < before:
            cur_date_and_time = cur_date_and_time - timedelta(hours=1)
            xi[j, 0] = int(cur_date_and_time.year == 2017)
            xi[j, 1] = float(cur_date_and_time.day) / month2days[str(cur_date_and_time.month)]
            xi[j, 2] = float(cur_date_and_time.month) / 12
            xi[j, 3] = float(cur_date_and_time.hour) / 24
            xi[j, 4] = cur_date_and_time.weekday() / 7
            j += 1
        X.append(xi)

    # ###########################read test data set#########################
    file = pd.read_csv('test.csv')
    df = pd.DataFrame(file)
    test_set = []
    for row in df.iterrows():
        # "date and time" in test set & create later time
        cur_date_and_time = datetime.strptime(row[1]["date"], "%d/%m/%Y %H:%M")  # string -> datetime
        xi = np.zeros(shape=(before, 5))  # features of one timestamp * 3

        # "date and time" in test set
        xi[0, 0] = int(cur_date_and_time.year == 2017)
        xi[0, 1] = float(cur_date_and_time.day) / month2days[str(cur_date_and_time.month)]
        xi[0, 2] = float(cur_date_and_time.month) / 12
        xi[0, 3] = float(cur_date_and_time.hour) / 24
        xi[0, 4] = cur_date_and_time.weekday() / 7

        j = 1
        while j < before:
            cur_date_and_time = cur_date_and_time - timedelta(hours=1)
            xi[j, 0] = int(cur_date_and_time.year == 2017)
            xi[j, 1] = float(cur_date_and_time.day) / month2days[str(cur_date_and_time.month)]
            xi[j, 2] = float(cur_date_and_time.month) / 12
            xi[j, 3] = float(cur_date_and_time.hour) / 24
            xi[j, 4] = cur_date_and_time.weekday() / 7
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


def create_data_add_weektime_period(train_prob=0.8):
    # read train data set
    file = pd.read_csv('train.csv')
    df = pd.DataFrame(file)
    X = []
    y = []
    for row in df.iterrows():
        _, date_and_time, speed = row[1]
        xi_list = re.split("/| ", date_and_time)  # len(xi_list)=4
        xi = np.zeros(shape=(7,))
        xi[0] = float(xi_list[0]) / month2days[xi_list[1]]
        xi[1] = float(xi_list[1]) / 12
        xi[2] = int(xi_list[2] == "2017")
        hour = float(xi_list[3].split(":")[0])
        xi[3] = hour / 24
        xi[4] = datetime.strptime("-".join(xi_list[0:3]), "%d-%m-%Y").weekday() / 7
        if 7 <= hour <= 9:  # 是否早高峰
            xi[5] = 1
        else:
            xi[5] = 0
        if 17 <= hour <= 19:
            xi[6] = 1
        else:
            xi[6] = 0
        X.append(xi)
        y.append(float(speed))

    # read test data set
    file = pd.read_csv('test.csv')
    df = pd.DataFrame(file)
    test_set = []
    for row in df.iterrows():
        _, date_and_time = row[1]
        xi_list = re.split("/| ", date_and_time)  # len(xi_list)=4
        xi = np.zeros(shape=(7,))
        xi[0] = float(xi_list[0]) / month2days[xi_list[1]]
        xi[1] = float(xi_list[1]) / 12
        xi[2] = int(xi_list[3] == "2017")
        hour = float(xi_list[3].split(":")[0])
        xi[3] = hour / 24
        xi[4] = datetime.strptime("-".join(xi_list[0:3]), "%d-%m-%Y").weekday() / 7
        if 7 <= hour <= 9:  # 是否早高峰
            xi[5] = 1
        else:
            xi[5] = 0
        if 17 <= hour <= 19:
            xi[6] = 1
        else:
            xi[6] = 0
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

    X_train = X[0: int(total * train_prob)]
    y_train = y[0: int(total * train_prob)]
    X_test = X[int(total * train_prob):]
    y_test = y[int(total * train_prob):]

    return X_train, y_train, X_test, y_test, test_set


def create_data_add_weektime(train_prob=0.8):
    # read train data set
    file = pd.read_csv('train.csv')
    df = pd.DataFrame(file)
    X = []
    y = []
    for row in df.iterrows():
        _, date_and_time, speed = row[1]
        xi_list = re.split("/| ", date_and_time)  # len(xi_list)=4
        xi = np.zeros(shape=(5,))
        xi[0] = float(xi_list[0]) / month2days[xi_list[1]]
        xi[1] = float(xi_list[1]) / 12
        xi[2] = int(xi_list[2] == "2017")
        xi[3] = float(xi_list[3].split(":")[0]) / 24
        xi[4] = datetime.strptime("-".join(xi_list[0:3]), "%d-%m-%Y").weekday() / 7
        X.append(xi)
        y.append(float(speed))

    # read test data set
    file = pd.read_csv('test.csv')
    df = pd.DataFrame(file)
    test_set = []
    for row in df.iterrows():
        _, date_and_time = row[1]
        xi_list = re.split("/| ", date_and_time)  # len(xi_list)=4
        xi = np.zeros(shape=(5,))
        xi[0] = float(xi_list[0]) / month2days[xi_list[1]]
        xi[1] = float(xi_list[1]) / 12
        xi[2] = int(xi_list[3] == "2017")
        xi[3] = float(xi_list[3].split(":")[0]) / 24
        xi[4] = datetime.strptime("-".join(xi_list[0:3]), "%d-%m-%Y").weekday() / 7
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

    X_train = X[0: int(total * train_prob)]
    y_train = y[0: int(total * train_prob)]
    X_test = X[int(total * train_prob):]
    y_test = y[int(total * train_prob):]

    return X_train, y_train, X_test, y_test, test_set


def create_data(train_prob=0.8):
    # read train data set
    file = pd.read_csv('train.csv')
    df = pd.DataFrame(file)
    X = []
    y = []
    for row in df.iterrows():
        _, date_and_time, speed = row[1]
        xi_list = re.split("/| ", date_and_time)  # len(xi_list)=4
        xi = np.zeros(shape=(4,))
        xi[0] = float(xi_list[0]) / month2days[xi_list[1]]
        xi[1] = float(xi_list[1]) / 12
        xi[2] = int(xi_list[2] == "2017")
        xi[3] = float(xi_list[3].split(":")[0]) / 24
        X.append(xi)
        y.append(float(speed))

    # read test data set
    file = pd.read_csv('test.csv')
    df = pd.DataFrame(file)
    test_set = []
    for row in df.iterrows():
        _, date_and_time = row[1]
        xi_list = re.split("/| ", date_and_time)  # len(xi_list)=4
        xi = np.zeros(shape=(4,))
        xi[0] = float(xi_list[0]) / month2days[xi_list[1]]
        xi[1] = float(xi_list[1]) / 12
        xi[2] = int(xi_list[3] == "2017")
        xi[3] = float(xi_list[3].split(":")[0]) / 24
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

    # write to test.csv
    file = pd.read_csv(r'test.csv')  # file.columns=获取列索引值
    file['speed'] = y_pred  # 将新列的名字设置为cha
    file.to_csv(r"test_submit_{model}.csv".format(model=model_name), mode='a')
    # mode=a，以追加模式写入,header表示列名，默认为true,index表示行名，默认为true，再次写入不需要行名


def main():
    X_train, y_train, X_test, y_test, test_set = create_data_add_weektime_ma_bachforth()

    rfr = RandomForestRegressor()
    rfr.fit(X_train, y_train)
    prec = rfr.score(X_test, y_test)  # 计算拟合曲线针对训练样本的拟合准确性
    print(prec)

    # feature_importances(rfr, ["date", "month", "year", "hour", "dayofweek", "Morning peak", "Evening peak"])

    # final_predict(rfr, "randomforest", test_set)


if __name__ == '__main__':
    main()
