# -*- coding: utf-8 -*-
"""
Created on Wed Mar 17 17:35:25 2021
@author: saniy
"""

import librosa
import librosa.display
import IPython.display as ipd
import matplotlib.pyplot as plt
import numpy as np

audio_file = "sound_leak/l10/h5-1.wav"

ipd.Audio(audio_file)

# load audio files with librosa
signal, sr = librosa.load(audio_file)

mfccs = librosa.feature.mfcc(y=signal, n_mfcc=13, sr=sr)
mfccs.shape


# plt.figure(figsize=(25, 10))
# librosa.display.specshow(mfccs, 
#                          x_axis="time", 
#                          sr=sr)
# plt.colorbar(format="%+2.f")
# plt.show()

delta_mfccs = librosa.feature.delta(mfccs)

delta2_mfccs = librosa.feature.delta(mfccs, order=2)

print(delta_mfccs.shape)

# plt.figure(figsize=(25, 10))
# librosa.display.specshow(delta_mfccs, 
#                          x_axis="time", 
#                          sr=sr)
# plt.colorbar(format="%+2.f")
# plt.show()

# plt.figure(figsize=(25, 10))
# librosa.display.specshow(delta2_mfccs, 
#                          x_axis="time", 
#                          sr=sr)
# plt.colorbar(format="%+2.f")
# plt.show()

mfccs_features = np.concatenate((mfccs, delta_mfccs, delta2_mfccs))

print(mfccs_features.shape)

def predict(model, X, y):

    # add a dimension to input data for sample - model.predict() expects a 4d array in this case
    X = X[np.newaxis, ...] # array shape (1, 130, 13, 1)

    # perform prediction
    prediction = model.predict(X)

    # get index with max value
    predicted_index = np.argmax(prediction, axis=1)

    print("Target: {}, Predicted label: {}".format(y, predicted_index))


# predict(model, X_to_predict, y_to_predict)
    new_output = model.predict(mfccs)
    print(new_output)


