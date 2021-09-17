import json
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Conv1D, BatchNormalization, MaxPool1D, Dropout, Dense, Bidirectional, GRU, Activation, add, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau

DATASET_PATH = "data_mfcc_nava.json"


def load_data(data_path):
  with open(data_path, "r") as fp:
    data = json.load(fp)

  x = np.array(data["mfcc"])
  y = np.array(data["labels"])
  return x, y



def prepare_datasets(test_size, validation_size):

  # load data
  X, Y = load_data(DATASET_PATH)

  # create train/test split
  X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size)

  # create train/validation split
  X_train, X_validation, Y_train, Y_validation = train_test_split(X_train, Y_train, test_size=validation_size)

  return X_train, X_validation, X_test, Y_train, Y_validation, Y_test



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


def Resnet_encoder(model, filter):

    x = model

    model = Conv1D(filters=filter, kernel_size=3, strides=2, padding='same')(model)
    model = Dropout(0.1)(model)
    model = BatchNormalization()(model)
    model = Activation('relu')(model)

    model = Conv1D(filters=int(filter/2), kernel_size=3, strides=1, padding='same')(model)
    model = Dropout(0.2)(model)
    model = BatchNormalization()(model)
    model = Activation('relu')(model)

    # expand channels for the sum
    shortcut = Conv1D(filters=int(filter/2), kernel_size=3, padding='same')(x)
    shortcut = BatchNormalization()(shortcut)

    output_block = add([shortcut, model])
    output_block = Activation('relu')(output_block)

    return output_block



def Encoder_CNN(model):

    model = Resnet_encoder(model, 128)

    model = Resnet_encoder(model, 64)

    model = Resnet_encoder(model, 32)

    return model

def Decoder_GRU(model):

    model = Bidirectional(GRU(16, return_sequences=True))(model)
    model = BatchNormalization()(model)

    model = Bidirectional(GRU(32, return_sequences=True))(model)
    model = BatchNormalization()(model)

    model = Bidirectional(GRU(64, return_sequences=True))(model)
    model = BatchNormalization()(model)

    model = Bidirectional(GRU(128))(model)
    model = BatchNormalization()(model)

    return model


def build_model(input_shape):

    # Input Layer
    input_layer = Input(shape=input_shape)

    # Resnet CNN Encoder
    model = Encoder_CNN(input_layer)

    # GRU Decoder
    model = Decoder_GRU(model)

    # FC
    model = Dense(16, activation='relu', name="FC_16")(model)
    model = Dropout(0.6)(model)

    # output layer
    output_layer = Dense(7, activation='softmax', name="FC_Output")(model)

    model = Model(inputs=input_layer, outputs=output_layer)

    return model


if __name__ == "__main__":
    # get train, validation , test splits
    X_train, X_validation, X_test, y_train, y_validation, y_test = prepare_datasets(0.15, 0.15)

    input_shape = (X_train.shape[1], X_train.shape[2])

    # create network
    model = build_model(input_shape)

    # compile the network
    optimizer = Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer,
                  loss="sparse_categorical_crossentropy",
                  metrics=['accuracy'])

    checkpoint_callback = ModelCheckpoint('weights.h5', monitor='val_accuracy', verbose=1,
                                          save_best_only=True, mode='max')

    reducelr_callback = ReduceLROnPlateau(monitor='val_accuracy',
                                          factor=0.7, patience=7,
                                          min_delta=0.01, verbose=1)

    callbacks_list = [checkpoint_callback, reducelr_callback]

    # plot_model(model, to_file="CNN_GRU_schema.png", show_shapes=True)
    model.summary()

    # train CNN

    history = model.fit(X_train, y_train, validation_data=(X_validation, y_validation), batch_size=32, epochs=250, callbacks=callbacks_list)
    plot_history(history)

    # evaluate model on test set
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2)
    print("\nTest Accuracy : ", test_acc)

