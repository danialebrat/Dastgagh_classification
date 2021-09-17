import json
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Dense, Dropout, Conv1D, BatchNormalization, Activation, Input, add, Flatten, MaxPool1D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau


DATA_PATH = "data_mfcc_nava.json"

def load_data(data_path):
  with open(data_path, "r") as fp:
    data = json.load(fp)

  X = np.array(data["mfcc"])
  y = np.array(data["labels"])
  return X, y


def plot_history(history):

  """ plot accuracy/loss for training/validation set as a function of the epochs"""

  fig, axs = plt.subplots(2)

  # create accuracy subplot

  axs[0].plot(history.history["accuracy"], label="train accuracy")
  axs[0].plot(history.history["val_accuracy"], label="test accuracy")
  axs[0].set_ylabel("Accuracy")
  axs[0].legend(loc="lower right")
  axs[0].set_title("Accuracy eval")

   # create error subplot

  axs[1].plot(history.history["loss"], label="train error")
  axs[1].plot(history.history["val_loss"], label="test error")
  axs[1].set_ylabel("Error")
  axs[1].set_xlabel("Epoch")
  axs[1].legend(loc="upper right")
  # axs[1].set_title("Accuracy eval")

  plt.show()



def prepare_datasets(test_size, validation_size):

  #load data
  X, y = load_data(DATA_PATH)
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= test_size)
  X_train, X_validation, y_train, y_validation = train_test_split(X_train, y_train, test_size= validation_size)


  return X_train, X_validation, X_test, y_train, y_validation, y_test


def Resnet_encoder(model, filter):

    x = model

    model = Conv1D(filters=filter, kernel_size=3, padding='same')(model)
    model = Dropout(0.1)(model)
    model = BatchNormalization()(model)
    model = Activation('relu')(model)

    model = Conv1D(filters=int(filter/2), kernel_size=3, padding='same')(model)
    model = Dropout(0.2)(model)
    model = BatchNormalization()(model)
    model = Activation('relu')(model)

    # expand channels for the sum
    shortcut = Conv1D(filters=int(filter/2), kernel_size=1, padding='same')(x)
    shortcut = BatchNormalization()(shortcut)

    output_block = add([shortcut, model])
    output_block = Activation('relu')(output_block)

    return output_block



def Resnet_decoder(model, filter):

    x = model

    model = Conv1D(filters=filter, kernel_size=3, padding='same')(model)
    model = Dropout(0.1)(model)
    model = BatchNormalization()(model)
    model = Activation('relu')(model)

    model = Conv1D(filters=int(filter*2), kernel_size=3, padding='same')(model)
    model = Dropout(0.2)(model)
    model = BatchNormalization()(model)
    model = Activation('relu')(model)

    # expand channels for the sum
    shortcut = Conv1D(filters=int(filter*2), kernel_size=1, padding='same')(x)
    shortcut = BatchNormalization()(shortcut)

    output_block = add([shortcut, model])
    output_block = Activation('relu')(output_block)

    return output_block



def build_model(input_shape):

    input_layer = Input(shape=input_shape)

    model = Resnet_encoder(input_layer, 128)

    model = Resnet_encoder(model, 64)

    model = Resnet_encoder(model, 32)

    model = Resnet_decoder(model, 16)

    model = Resnet_decoder(model, 32)

    model = Resnet_decoder(model, 64)

    gap_layer = MaxPool1D(pool_size=3, strides=1)(model)

    bottleneck = Dense(16)(gap_layer)
    bottleneck = Dropout(0.6)(bottleneck)

    flatten = Flatten()(bottleneck)

    output_layer = Dense(7, activation='softmax')(flatten)

    model = Model(inputs=input_layer, outputs=output_layer)

    return model


if __name__ == "__main__":
    # get train, validation , test splits
    X_train, X_validation, X_test, y_train, y_validation, y_test = prepare_datasets(0.15, 0.15)

    # create network
    input_shape = (X_train.shape[1], X_train.shape[2])
    model = build_model(input_shape)

    # compile model
    optimizer = Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer,
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    checkpoint_callback = ModelCheckpoint('weights.h5', monitor='val_accuracy', verbose=1,
                                          save_best_only=True, mode='max')

    reducelr_callback = ReduceLROnPlateau(
        monitor='val_accuracy', factor=0.7, patience=7, min_delta=0.01,
        verbose=1
    )

    callbacks_list = [checkpoint_callback, reducelr_callback]

    model.summary()

    # train model6
    history = model.fit(X_train, y_train, validation_data=(X_validation, y_validation), batch_size=32, epochs=200,
                        callbacks=callbacks_list)

    # plot accuracy
    plot_history(history)

    # evaluate model on test set
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2)
    print("\nTest Accuracy : ", test_acc)