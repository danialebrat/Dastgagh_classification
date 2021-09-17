import os
import librosa
import math
import json
import numpy as np


DATASET_PATH = "E:/Danial.ebrat_97112113/Dataset/SuperMini Dataset"
JSON_PATH = "data_mfcc_supermini.json"



# dictionary to store data
data = {
      "mapping": [],
      "mfcc": [],
      "labels": []
  }

SAMPLE_RATE = 22050
HOP_LENGTH = 512
N_FFT = 2048
N_MFCC = 24

DURATION = 20 #second
SAMPLES_PER_SEGMENT = SAMPLE_RATE * DURATION
expected_num_mfcc_vectors_per_segment = math.ceil(SAMPLES_PER_SEGMENT / HOP_LENGTH)

def save_to_json(json_path, data):

  with open(json_path, "w") as fp:
    json.dump(data, fp, indent=4)


def calculate_mfcc(signal, sr=SAMPLE_RATE, hop_length=HOP_LENGTH, n_fft=N_FFT, n_mfcc=N_MFCC):
    mfcc = librosa.feature.mfcc(signal,
                                sr=sr,
                                n_fft=n_fft,
                                n_mfcc=n_mfcc,
                                hop_length=hop_length)

    mfcc = mfcc.T

    return mfcc


def store_mfcc(filename, seg_num, mfcc, label):

  # print("mfcc shape {}, expected {}".format(mfcc.shape, expected_num_mfcc_vectors_per_segment ))
  if len(mfcc) == expected_num_mfcc_vectors_per_segment:
      data["mfcc"].append(mfcc.tolist())
      data["labels"].append(label)
      print("{}, segment:{} is labeled to {}".format(filename, seg_num, label))



def number_of_segments(dirpath, f):

    file_path = os.path.join(dirpath, f)
    signal, sr = librosa.load(str(file_path), sr= SAMPLE_RATE)
    duration = librosa.get_duration(signal, sr=SAMPLE_RATE, n_fft=N_FFT, hop_length=HOP_LENGTH)
    SAMPLES_PER_TRACK = SAMPLE_RATE * duration
    num_segments = int(SAMPLES_PER_TRACK / SAMPLES_PER_SEGMENT)

    return signal, num_segments


def save_semantic_label(dirpath):

  dirpath_components = dirpath.split("/")
  semantic_label = dirpath_components[-1]
  data["mapping"].append(semantic_label)
  print("\n Processing {}".format(semantic_label))


def get_label(f):

  label = int(f[2])

  if label == 0 or label == 1 or label == 2:
    pass
  else:
    label = label -1

  return label


def save_mfcc(dataset_path):
    # loop through all the Dastgahs
    for i, (dirpath, dirnames, filenames) in enumerate(os.walk(dataset_path)):

        # ensure we're not at root level
        if dirpath is not dataset_path:

            # save the semantic label
            save_semantic_label(dirpath)

            # process titles for specific Dastgah
            for f in filenames:

                # get duration
                signal, num_segments = number_of_segments(dirpath, f)
                label = get_label(f)

                # process segments extracting mfcc and storing data
                for s in range(num_segments):
                    start_sample = SAMPLES_PER_SEGMENT * s
                    finish_sample = start_sample + SAMPLES_PER_SEGMENT

                    mfcc = calculate_mfcc(signal[start_sample:finish_sample])

                    # store mfcc for segment if it has the expected length
                    store_mfcc(f, s, mfcc, label)


if __name__ == "__main__":
    save_mfcc(DATASET_PATH)
    save_to_json(JSON_PATH, data)




