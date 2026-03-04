import os
import time
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras
from tensorflow.data import Dataset
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, Dense, Flatten, BatchNormalization, Conv2D, MaxPool2D, Dropout, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint
# from sklearn.metrics import confusion_matrix



def main():
    (train_images, train_labels), (test_images, test_labels) = keras.datasets.cifar10.load_data()

    CLASS_NAMES = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]

    validation_images, validation_labels = train_images[:5000], train_labels[:5000]
    train_images, train_labels = train_images[5000:], train_labels[5000:]

    train_ds = Dataset.from_tensor_slices((train_images, train_labels))
    validation_ds = Dataset.from_tensor_slices((validation_images, validation_labels))
    test_ds = Dataset.from_tensor_slices((test_images, test_labels))


    # plt.figure(figsize=(20,20))
    # for i, (image, label) in enumerate(train_ds.take(9)):
    #     plt.subplot(3, 3, i+1)
    #     plt.imshow(image)
    #     plt.title(CLASS_NAMES[label.numpy()[0]])
    #     plt.axis("off")

    # plt.show()

    # train_ds_size = tf.data.experimental.cardinality(train_ds).numpy()
    # validation_ds_size = tf.data.experimental.cardinality(validation_ds).numpy()
    # test_ds_size = tf.data.experimental.cardinality(test_ds).numpy()

    # print("train data size:", train_ds_size)
    # print("validation data size:", validation_ds_size)
    # print("test data size:", test_ds_size)

    train_ds = (train_ds
                    .map(process_images)
                    .shuffle(buffer_size=1000, seed=42, reshuffle_each_iteration=True)
                    .batch(batch_size=32, drop_remainder=True)
                    .prefetch(tf.data.AUTOTUNE))
    validation_ds = (validation_ds
                        .map(process_images)
                        .shuffle(buffer_size=1000, seed=42, reshuffle_each_iteration=True)
                        .batch(batch_size=32, drop_remainder=True)
                        .prefetch(tf.data.AUTOTUNE))
    test_ds = (test_ds
                    .map(process_images)
                    .shuffle(buffer_size=1000, seed=42, reshuffle_each_iteration=True)
                    .batch(batch_size=32, drop_remainder=True)
                    .prefetch(tf.data.AUTOTUNE))

    # model = Sequential([
    #     Conv2D(filters=96, kernel_size=(11, 11), strides=(4, 4), activation="relu", input_shape=(227, 227, 3)),
    #     BatchNormalization(),
    #     MaxPool2D(pool_size=(3,3), strides=(2,2)),

    #     Conv2D(filters=256, kernel_size=(5, 5), strides=(1, 1), activation="relu", padding="same"),
    #     BatchNormalization(),
    #     MaxPool2D(pool_size=(3, 3), strides=(2, 2)),

    #     Conv2D(filters=384, kernel_size=(3, 3), strides=(1, 1), activation="relu", padding="same"),
    #     BatchNormalization(),

    #     Conv2D(filters=384, kernel_size=(3, 3), strides=(1, 1), activation="relu", padding="same"),
    #     BatchNormalization(),

    #     Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), activation="relu", padding="same"),
    #     BatchNormalization(),
    #     MaxPool2D(pool_size=(3, 3), strides=(2, 2)),

    #     Flatten(),
    #     Dense(4096, activation="relu"),
    #     Dropout(0.5),
    #     Dense(4096, activation="relu"),
    #     Dropout(0.5),
    #     Dense(10, activation="softmax")
    # ])

    model = Sequential([
        Conv2D(64, (3,3), padding="same", activation="relu", input_shape=(32,32,3)),
        BatchNormalization(),
        MaxPool2D((2,2)),

        Conv2D(128, (3,3), padding="same", activation="relu"),
        BatchNormalization(),
        MaxPool2D((2,2)),

        Conv2D(256, (3,3), padding="same", activation="relu"),
        BatchNormalization(),
        MaxPool2D((2,2)),

        GlobalAveragePooling2D(),
        Dense(512, activation="relu"),
        Dropout(0.5),
        Dense(10, activation="softmax")
    ])

    run_logdir = get_run_logdir()
    tensorboard_cb = TensorBoard(run_logdir)

    model.compile(optimizer=Adam(learning_rate=0.0001), loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    model.summary()

    model.fit(train_ds, validation_data=validation_ds, validation_freq=1, epochs=30, verbose=2, callbacks=[tensorboard_cb])

    model.evaluate(test_ds)


def process_images(image, label):
    image = tf.image.per_image_standardization(image)
    # image = tf.image.resize(image, (227, 227))
    return image, label


def get_run_logdir():
    current_time = time.strftime("%Y_%m_%d-%H_%M_%S")
    log_dir = os.path.join("logs", current_time)
    return log_dir


if __name__=="__main__":
    main()