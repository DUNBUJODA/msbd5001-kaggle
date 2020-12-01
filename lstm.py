import numpy as np
import re
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import tensorflow as tf  # env: tf2
from tensorflow import keras
from tensorflow.keras import layers

month2days = {"1": 31, "2": 28, "3": 31, "4": 30, "5": 31, "6": 30, "7": 31, "8": 31, "9": 30, "10": 31, "11": 30,
              "12": 31}

before = 14
maxlen = 14


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
    X = X.astype('float32').reshape((-1, before * 5, 1))
    test_set = test_set.astype('float32').reshape((-1, before * 5, 1))

    return X, y, test_set


def build_model():
    # model
    Model = keras.Sequential()
    Model.add(layers.LSTM(units=50, return_sequences=True, input_shape=(before * 5, 1)))
    Model.add(layers.Dropout(0.2))
    Model.add(layers.Dense(units=1))

    # compile & training
    Model.compile(optimizer="adam",
                  loss=keras.losses.MeanSquaredError(),
                  metrics=['mse'])

    return Model


def create_data():
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
        hour = float(xi_list[3].split(":")[0])
        xi[3] = hour / 24
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
        hour = float(xi_list[3].split(":")[0])
        xi[3] = hour / 24
        xi[4] = datetime.strptime("-".join(xi_list[0:3]), "%d-%m-%Y").weekday() / 7
        test_set.append(xi)

    # to array
    X = np.array(X, dtype=float)
    y = np.array(y, dtype=float)
    test_set = np.array(test_set, dtype=float)
    X = X.astype('float32').reshape((-1, 5))
    test_set = test_set.astype('float32').reshape((-1, 5))

    return X, y, test_set


def create_data_add_weektime_period():
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
    X = X.astype('float32').reshape((-1, 7))
    test_set = test_set.astype('float32').reshape((-1, 7))

    return X, y, test_set


def final_predict(test_set, model_name):
    restored_model = build_model()
    restored_model = tf.keras.models.load_model('keras_model.hdf5')
    y_pred = restored_model.predict(test_set)
    y_pred = y_pred.reshape(y_pred.shape[0])
    file = pd.read_csv(r'test.csv')  # file.columns=获取列索引值
    file['speed'] = y_pred  # 将新列的名字设置为cha
    file.to_csv(r"test_submit_{model}.csv".format(model=model_name), mode='a')


def train_and_compile(trainDataset, valDataset):
    mlpModel = build_model()

    callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=100, monitor='val_loss'),
        tf.keras.callbacks.ModelCheckpoint('keras_model.hdf5', save_best_only=True, monitor='val_loss', mode='min')
    ]
    history = mlpModel.fit(trainDataset, epochs=1000, validation_data=valDataset.batch(20), callbacks=callbacks)

    # history record the accuracy/loss on training set/validation set after every epoch
    plt.plot(history.history['mse'])
    plt.plot(history.history['val_mse'])
    plt.legend(['training', 'valivation'], loc='upper left')
    plt.show()
    loss, mse = mlpModel.evaluate(valDataset.batch(100))
    print("loss={loss}, mse={mse}".format(loss=loss, mse=mse))


def reload_and_evaluate(valDataset):
    restored_model = build_model()
    loss, mse = restored_model.evaluate(valDataset.batch(5))
    print("loss={loss}, mse={mse}".format(loss=loss, mse=mse))

    restored_model = tf.keras.models.load_model('keras_model.hdf5')
    loss, mse = restored_model.evaluate(valDataset.batch(5))
    print("loss={loss}, mse={mse}".format(loss=loss, mse=mse))


def main():
    # data
    X, y, test_set = create_data_add_weektime_ma()
    allDataset = tf.data.Dataset.from_tensor_slices((X, y)).shuffle(buffer_size=100, reshuffle_each_iteration=False)
    trainDataset = allDataset.take(int(X.shape[0] * 0.8)).batch(20)
    valDataset = allDataset.skip(int(X.shape[0] * 0.8))

    train_and_compile(trainDataset, valDataset)
    # reload_and_evaluate(valDataset)

    # final_predict(test_set, "mlp")


if __name__ == '__main__':
    main()
