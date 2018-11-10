import numpy as np
import os
import pandas as pd
import sklearn
import tensorflow as tf
from keras import Sequential
from sklearn.model_selection import train_test_split
from keras.utils import np_utils
from keras.layers import Dense, Activation, Dropout
from keras.layers import Conv2D
from keras.layers.pooling import MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.layers.core import Flatten
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
from keras.preprocessing.image import array_to_img, img_to_array, load_img
from keras import optimizers
from keras import callbacks
from keras import regularizers
from keras.optimizers import Adam
from keras.models import load_model
from PIL import Image
import cv2

accuracy = 0
EPOCHS = 10
BATCH_SIZE_TRAIN = 32
BATCH_SIZE_TEST = 32
INPUT_WIDTH = 128
INPUT_HEIGHT = 128
CHANNELS = 3
LOADING_PATH = './train_save_pic/'
sgd = optimizers.SGD(lr=0.01, momentum=0.9, nesterov=True)
adadelta = optimizers.Adadelta(lr=0.50, rho=0.95, epsilon=1e-06)
datagen = ImageDataGenerator(
    featurewise_center=True,
    featurewise_std_normalization=True,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True)


def read_image(img_name):
    im = cv2.imread(img_name, 1)
    im = cv2.resize(im, (32, 32), interpolation=cv2.INTER_NEAREST)
    data = np.array(im)
    return data


def load_data():
    images = []
    files = os.listdir('./train_save_pic')
    files.sort(key=lambda x: int(x[18:22]))
    for fn in files:
        print(fn[18:22])
        fd = os.path.join('./train_save_pic', fn)
        images.append(read_image(fd))

    print('load success!')
    X = np.array(images)
    y = np.loadtxt('result.txt')
    return X, y


if __name__ == '__main__':
    # load data
    X, y = load_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=30)
    print(X_train.shape)
    print(X_test.shape)
    print(y_train.shape)
    print(y_test.shape)

    X_train = X_train.reshape(-1, 32, 32, 3)
    X_test = X_test.reshape(-1, 32, 32, 3)
    y_train = np_utils.to_categorical(y_train, num_classes=2)
    y_test = np_utils.to_categorical(y_test, num_classes=2)

    print(X_train.shape)
    print(X_test.shape)
    print(y_train.shape)
    print(y_test.shape)

    print("Changing succeeded!")

    model = load_model('opencv_classifier_21_cicle_961.h5')
    # 评估模型
    loss, accuracy = model.evaluate(X_train, y_train)
    
    print('\ntest loss', loss)
    print('accuracy', accuracy)
    loss, accuracy = model.evaluate(X_test, y_test)
    print('\ntest loss', loss)
    print('accuracy', accuracy)
    '''
    model = Sequential()
    # conv layer 1 as follows
    model.add(Conv2D(nb_filter=96,nb_row=3,nb_col=3,border_mode='same',input_shape=(32, 32, 3),activation='relu',kernel_regularizer=regularizers.l2(0.01)))
    model.add(Dropout(0.5))

    # pooling layer 1 as follows
    model.add(MaxPooling2D(
        pool_size=(2, 2),
        strides=(2, 2),
        border_mode='same'))
    model.add(BatchNormalization())
    # conv layer 2 as follows
    # model.add(Conv2D(256, 3, 3, border_mode='same', activation='relu'))
    model.add(Conv2D(256, 3, 3, border_mode='same', activation='relu'))
    model.add(Dropout(0.5))
    # model.add(Conv2D(128, 3, 3, border_mode='same', activation='relu'))

    # pooling layer 2 as follows
    # model.add(Conv2D(256, 3, 3, border_mode='same', activation='relu'))
    # model.add(Dropout(0.25))
    model.add(Conv2D(64, 3, 3, border_mode='same', activation='relu'))
    model.add(MaxPooling2D(2, 2, border_mode='same'))

    ########################
    model.add(Flatten())
    # model.add(Dense(128, activation='relu'))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    # model.add(Dense(128, activation='relu'))
    # model.add(Dense(1024, activation='relu'))
    # model.add(Dense(1024, activation='relu'))

    ########################
    model.add(Dense(2, activation='softmax'))

    ########################
    adam = Adam(lr=1e-4)

    ########################
    model.compile(optimizer=adadelta,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    '''
    print('Training ------------')
    # Another way to train the model
    # datagen.fit(X_train)
    model.fit(X_train, y_train, epochs=3, batch_size=64, )

    print('\nTesting ------------')
    # Evaluate the model with the metrics we defined earlier
    loss, accuracy = model.evaluate(X_test, y_test)

    print('\ntest loss: ', loss)
    print('\ntest accuracy: ', accuracy)

    model.save('opencv_classifier_21.h5')  # HDF5文件，pip install h5py
    print('\nSuccessfully saved as opencv_classifier_21.h5')
