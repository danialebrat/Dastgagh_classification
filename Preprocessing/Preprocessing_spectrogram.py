import os
import librosa
import math
import json
import numpy as np


DATASET_PATH = "E:/Danial.ebrat_97112113/Dataset/Nava"
JSON_PATH = "data_spectrogram"



# dictionary to store data
data = {
      "mapping": [],
      "spectrogram": [],
      "labels": []
  }

SAMPLE_RATE = 22050
HOP_LENGTH = 512
N_FFT = 2048

DURATION = 20 #second
SAMPLES_PER_SEGMENT = SAMPLE_RATE * DURATION
expected_num_vectors_per_segment = math.ceil(SAMPLES_PER_SEGMENT / HOP_LENGTH)

def save_to_json(json_path, data):

  with open(json_path, "w") as fp:
    json.dump(data, fp, indent=4)



def calculate_spectrogram(signal, n_fft=N_FFT, hop_length=HOP_LENGTH):
    spectrogram = librosa.stft(signal,
                               n_fft=n_fft,
                               hop_length=hop_length,
                               )

    spectrogram = np.abs(spectrogram) ** 2
    spectrogram = librosa.power_to_db(spectrogram)
    spectrogram = spectrogram.T

    return spectrogram


def store_spectrogram(filename, seg_num, spectrogram, label):

  if len(spectrogram) == expected_num_vectors_per_segment:
      data["spectrogram"].append(spectrogram.tolist())
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


def save_spectrogram(dataset_path):
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

                # process segments extracting spectrogram and storing data
                for s in range(num_segments):
                    start_sample = SAMPLES_PER_SEGMENT * s
                    finish_sample = start_sample + SAMPLES_PER_SEGMENT

                    spectrogram = calculate_spectrogram(signal[start_sample:finish_sample])

                    # store log_mel for segment if it has the expected length
                    store_spectrogram(f, s, spectrogram, label)




if __name__ == "__main__":

  save_spectrogram(DATASET_PATH)
  save_to_json(JSON_PATH, data)


