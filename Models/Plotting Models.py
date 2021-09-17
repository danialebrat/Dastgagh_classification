import json
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, \
    f1_score, classification_report
import seaborn as sns
import matplotlib.pyplot as plt


# DATA_PATH = "data_mfcc_supermini.json"

x_test = np.load("x_test.npy")
y_test = np.load("y_test.npy")
model = load_model("weights.h5")


def load_data(data_path):
  with open(data_path, "r") as fp:
    data = json.load(fp)
  x = np.array(data["mfcc"])
  y = np.array(data["labels"])

  return x,y


def accuracy_performance(y_test, y_pred):


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

    fig = plt.figure(figsize=(10,10))
    plt.clf()
    plt.title('Confusion Matrix')

    cmap = sns.cubehelix_palette(light=1, as_cmap=True)

    group_counts = ["{0:0.0f}".format(value) for value in cm.flatten()]
    # group_percentages = ["{0:.2%}".format(value) for value in cm.flatten()/np.sum(cm)]
    labels = [f"{v1}" for v1 in group_counts]
    labels = np.asarray(labels).reshape(7,7)

    x_axis_labels = dastgah # labels for x-axis
    y_axis_labels = dastgah # labels for y-axis

    # create seaborn heatmap with required labels
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
    # x, y = load_data(DATA_PATH)
    dastgah = ['Shur', 'Segah', 'Mahur', 'Homa', 'Rast', 'Nava', 'Char']

    # evaluate model on dataset
    classification_performance(x_test, y_test, dastgah, model)
