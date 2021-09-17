import json
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from tensorflow.keras.layers import add , Input, Activation, Dense, Conv1D, MaxPool1D, BatchNormalization, Dropout, Flatten, Bidirectional, LSTM, GRU
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

  plt.show()



def prepare_datasets(test_size, validation_size):

  #load data
  X, y = load_data(DATA_PATH)
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= test_size)
  X_train, X_validation, y_train, y_validation = train_test_split(X_train, y_train, test_size= validation_size)


  return X_train, X_validation, X_test, y_train, y_validation, y_test



def ResBlock(x , filters, return_sequences):

    r = Bidirectional(LSTM(filters, return_sequences=True))(x)
    r = BatchNormalization()(r)

    r = Bidirectional(LSTM(int(filters/2), return_sequences=return_sequences))(r)
    r = BatchNormalization()(r)

    if x.shape[-1] == filters:
        shortcut = x
    else:
        shortcut = Bidirectional(GRU(int(filters / 2), return_sequences=return_sequences))(x)  # shortcut (shortcut)

    o = add([r, shortcut])
    o = Activation('relu')(o) #Activation function

    return o


def decoder(x):

    x = Bidirectional(GRU(16, return_sequences=True))(x)
    x = BatchNormalization()(x)

    x = Bidirectional(GRU(32, return_sequences=True))(x)
    x = BatchNormalization()(x)

    x = Bidirectional(GRU(64, return_sequences=True))(x)
    x = BatchNormalization()(x)

    x = Bidirectional(GRU(128))(x)
    x = BatchNormalization()(x)

    return x

def build_model(input_shape):

    inputs = Input(input_shape)

    x = ResBlock(inputs, filters=128, return_sequences=True)
    x = ResBlock(x, filters=64, return_sequences=True)
    x = ResBlock(x, filters=32, return_sequences=True)

    x = decoder(x)


    x = Dense(16, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(7, activation='softmax')(x)

    model = Model(inputs, x)

    return model


if __name__ == "__main__":
    # get train, validation , test splits
    X_train, X_validation, X_test, y_train, y_validation, y_test = prepare_datasets(0.15, 0.15)

    # create network
    input_shape = (X_train.shape[1], X_train.shape[2])

    # build_model
    model = build_model(input_shape)

    # compile model
    optimizer = Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer,
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    model.summary()

    checkpoint_callback = ModelCheckpoint('weights.h5', monitor='val_accuracy', verbose=1,
                                          save_best_only=True, mode='max')

    reducelr_callback = ReduceLROnPlateau(
        monitor='val_accuracy', factor=0.7, patience=10, min_delta=0.01,
        verbose=1
    )

    callbacks_list = [checkpoint_callback, reducelr_callback]

    model.summary()

    # train CNN

    history = model.fit(X_train, y_train, validation_data=(X_validation, y_validation), batch_size=32, epochs=250,
                        callbacks=callbacks_list)
    plot_history(history)

    # plot accuracy
    plot_history(history)

    # evaluate model on test set
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2)
    print("\nTest Accuracy : ", test_acc)


