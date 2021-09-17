import json
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dense, Dropout, LSTM, Bidirectional, GRU, BatchNormalization
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, \
    f1_score, classification_report
import seaborn as sns


DATA_PATH = "data_mfcc_nava.json"

def load_data(data_path):
    with open(data_path, "r") as fp:
        data = json.load(fp)
    x = np.array(data["mfcc"])
    y = np.array(data["labels"])
    return x, y


def prepare_datasets(test_size, validation_size):
    # load data
    x, y = load_data(DATA_PATH)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size)
    x_train, x_validation, y_train, y_validation = train_test_split(x_train, y_train, test_size=validation_size)

    return x_train, x_validation, x_test, y_train, y_validation, y_test


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
    # axs[1].set_title("\nAccuracy eval")

    plt.savefig('Accuracy eval.png', dpi=100, bbox_inches='tight')
    plt.show()


def build_model(input_shape):

    # build network topology
    Autoencoder = Sequential()

    # 4 LSTM layers
    Autoencoder.add(Bidirectional(LSTM(128, input_shape=input_shape, return_sequences=True), input_shape=input_shape))
    Autoencoder.add(BatchNormalization())

    Autoencoder.add(Bidirectional(LSTM(64, return_sequences=True)))
    Autoencoder.add(BatchNormalization())

    Autoencoder.add(Bidirectional(LSTM(32, return_sequences=True)))
    Autoencoder.add(BatchNormalization())

    Autoencoder.add(Bidirectional(GRU(16, return_sequences=True)))
    Autoencoder.add(BatchNormalization())

    Autoencoder.add(Bidirectional(GRU(32, return_sequences=True)))
    Autoencoder.add(BatchNormalization())

    Autoencoder.add(Bidirectional(GRU(64, return_sequences=True)))
    Autoencoder.add(BatchNormalization())

    Autoencoder.add(Bidirectional(GRU(128)))
    Autoencoder.add(BatchNormalization())

    # dense layer
    Autoencoder.add(Dense(16, activation='relu'))
    Autoencoder.add(Dropout(0.6))

    # output layer
    Autoencoder.add(Dense(7, activation='softmax'))

    return Autoencoder


def accuracy_performance(y_test, y_pred):


    #importing accuracy_score, precision_score, recall_score, f1_score

    print('\nAccuracy: {:.2f}\n'.format(accuracy_score(y_test, y_pred)))

    print('Micro Precision: {:.2f}'.format(precision_score(y_test, y_pred, average='micro')))
    print('Micro Recall: {:.2f}'.format(recall_score(y_test, y_pred, average='micro')))
    print('Micro F1-score: {:.2f}\n'.format(f1_score(y_test, y_pred, average='micro')))

    print('Macro Precision: {:.2f}'.format(precision_score(y_test, y_pred, average='macro')))
    print('Macro Recall: {:.2f}'.format(recall_score(y_test, y_pred, average='macro')))
    print('Macro F1-score: {:.2f}\n'.format(f1_score(y_test, y_pred, average='macro')))

    print('Weighted Precision: {:.2f}'.format(precision_score(y_test, y_pred, average='weighted')))
    print('Weighted Recall: {:.2f}'.format(recall_score(y_test, y_pred, average='weighted')))
    print('Weighted F1-score: {:.2f}'.format(f1_score(y_test, y_pred, average='weighted')))



def plotting_cm(cm, dastgah):

    fig = plt.figure(figsize=(8,8))
    plt.clf()
    plt.title('Confusion Matrix')

    cmap = sns.cubehelix_palette(light=1, as_cmap=True)

    group_counts = ["{0:0.0f}".format(value) for value in cm.flatten()]
    # group_percentages = ["{0:.2%}".format(value) for value in cm.flatten()/np.sum(cm)]
    labels = [f"{v1}" for v1 in group_counts]
    labels = np.asarray(labels).reshape(7,7)

    x_axis_labels = dastgah # labels for x-axis
    y_axis_labels = dastgah # labels for y-axis

    # create seabvorn heatmap with required labels
    sns.heatmap(cm, xticklabels=x_axis_labels, yticklabels=y_axis_labels, annot=labels, fmt='', cmap=cmap)
    plt.savefig('confusion_matrix.png', dpi=100, bbox_inches='tight')
    plt.show()


def classification_performance(x, y, dastgah, model):

    prediction = np.argmax(model.predict(x), axis=-1)

    cm = confusion_matrix(y, prediction)

    # Calculate the confusion matrix

    accuracy_performance(prediction, y)

    print('\nClassification Report\n')
    print(classification_report(prediction, y, target_names=dastgah))

    plotting_cm(cm, dastgah)


if __name__ == "__main__":
    # get train, validation , test splits
    x_train, x_validation, x_test, y_train, y_validation, y_test = prepare_datasets(0.05, 0.05)

    np.save("x_test", x_test)
    np.save("y_test", y_test)

    # create network
    input_shape = (x_train.shape[1], x_train.shape[2])
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

    # train model
    history = model.fit(x_train, y_train, validation_data=(x_validation, y_validation), batch_size=32, epochs=250,
                        callbacks=callbacks_list)

    # plot accuracy
    plot_history(history)

    # evaluate model on test set
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
    print("\nTest Accuracy : ", test_acc)

    dastgah = ['Shur', 'Segah', 'Mahur', 'Homayoun', 'RastPanjgah', 'Nava', 'Chahargah']

    # evaluate model on dataset
    classification_performance(x_test, y_test, dastgah, model)
