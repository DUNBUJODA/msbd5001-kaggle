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


def create_data():
    """
    "several hours before" is considered as a complete feature for a sample
    these hours are continued, controlled by the BEFORE variable

    X sample:
    [year0, day0, month0, hour0, weekday0,
    year1, day1, month1, hour1, weekday1,
    year2, day2, month2, hour2, weekday2,
    ...]

    y sample: speed

    :param train_prob:
    :return:
    """
    # ####################read train data set######################
    file = pd.read_csv('train.csv')
    df = pd.DataFrame(file)
    X = []
    y = []
    # before = 14

    for row in df.iterrows():
        # "date and time" in train set & create later time
        cur_date_and_time = datetime.strptime(row[1]["date"], "%d/%m/%Y %H:%M")  # string -> datetime
        speed = row[1]["speed"]
        xi = np.zeros(shape=(before * 5,))  # features of one timestamp * 3

        # "date and time" in train set
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
        xi = np.zeros(shape=(before * 5,))  # features of one timestamp * 3

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
    """
    build a cnn model
    :return:
    """
    # model
    Model = keras.Sequential()
    Model.add(layers.Conv1D(128, 3, activation='relu'))
    Model.add(layers.GlobalAveragePooling1D())
    Model.add(layers.Dense(64, activation='relu'))
    Model.add(layers.Dense(1))

    # compile & training
    Model.compile(optimizer="adam",
                  loss=keras.losses.MeanSquaredError(),
                  metrics=['mse'])

    return Model


def train_and_compile(trainDataset, valDataset):
    """
    run the model
    :param trainDataset:
    :param valDataset:
    :return:
    """
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
    loss, mse = mlpModel.evaluate(valDataset.batch(5))
    print("loss={loss}, mse={mse}".format(loss=loss, mse=mse))


def main():
    # data
    X, y, test_set = create_data()
    allDataset = tf.data.Dataset.from_tensor_slices((X, y)).shuffle(buffer_size=100, reshuffle_each_iteration=False)
    trainDataset = allDataset.take(int(X.shape[0] * 0.8)).batch(20)
    valDataset = allDataset.skip(int(X.shape[0] * 0.8))

    train_and_compile(trainDataset, valDataset)
    # reload_and_evaluate(valDataset)
    # final_predict(test_set, "mlp")


if __name__ == '__main__':
    main()
