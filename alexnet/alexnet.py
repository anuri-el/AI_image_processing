import os
import time

import tensorflow as tf
from tensorflow import keras
from tensorflow.data import Dataset
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, Dense, Flatten, BatchNormalization, Conv2D, MaxPool2D, Dropout, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint


def main():
    (train_images, train_labels), (validation_images, validation_labels), (test_images, test_labels) = get_data()
    train_ds, validation_ds, test_ds = get_ds(train_images, train_labels, validation_images, validation_labels, test_images, test_labels)

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
    checkpoint_cb = ModelCheckpoint("alexnet_best.keras", monitor="val_accuracy", save_best_only=True, verbose=1)

    model.compile(optimizer=Adam(learning_rate=0.0001), loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    model.summary()

    model.fit(train_ds, validation_data=validation_ds, validation_freq=1, epochs=30, callbacks=[tensorboard_cb, checkpoint_cb])

    # model.evaluate(test_ds)
    model.save("alexnet_final.keras")


def get_data():
    (train_images, train_labels), (test_images, test_labels) = keras.datasets.cifar10.load_data()

    validation_images, validation_labels = train_images[:5000], train_labels[:5000]
    train_images, train_labels = train_images[5000:], train_labels[5000:]

    return (train_images, train_labels), (validation_images, validation_labels), (test_images, test_labels) 


def get_ds(train_images, train_labels, validation_images, validation_labels, test_images, test_labels):
    train_ds = Dataset.from_tensor_slices((train_images, train_labels))
    validation_ds = Dataset.from_tensor_slices((validation_images, validation_labels))
    test_ds = Dataset.from_tensor_slices((test_images, test_labels))

    train_ds = (train_ds
                    .map(process_images)
                    .shuffle(buffer_size=1000, seed=42, reshuffle_each_iteration=True)
                    .batch(batch_size=32)
                    .prefetch(tf.data.AUTOTUNE))
    validation_ds = (validation_ds
                        .map(process_images)
                        .batch(batch_size=32)
                        .prefetch(tf.data.AUTOTUNE))
    test_ds = (test_ds
                    .map(process_images)
                    .batch(batch_size=32)
                    .prefetch(tf.data.AUTOTUNE))
    
    return train_ds, validation_ds, test_ds


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