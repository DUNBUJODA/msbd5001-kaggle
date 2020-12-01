import numpy as np
import re
from sklearn.neighbors import KNeighborsRegressor  # env: tf4
import pandas as pd
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
    before = 30

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


def main():
    X_train, y_train, X_test, y_test, test_set = create_data_add_weektime_ma()

    k = 7
    knn = KNeighborsRegressor(k)
    knn.fit(X_train, y_train)
    prec = knn.score(X_test, y_test)  # 计算拟合曲线针对训练样本的拟合准确性
    print(prec)

    # y_pred = knn.predict(test_set)
    #
    # # write to test.csv
    # file = pd.read_csv(r'test.csv')  # file.columns=获取列索引值
    # file['speed'] = y_pred  # 将新列的名字设置为cha
    # file.to_csv(r"test_submit.csv", mode='a', header="speed")
    # # mode=a，以追加模式写入,header表示列名，默认为true,index表示行名，默认为true，再次写入不需要行名


if __name__ == '__main__':
    main()
