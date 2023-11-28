from google.colab import drive
drive.mount('/content/drive') #importing content from google drive

import os
from os import listdir
from os.path import isfile, join
from matplotlib import pyplot as plt
import scipy.io.wavfile as wav
import numpy as np
from keras.preprocessing import image
from keras.utils.np_utils import to_categorical


import matplotlib.pyplot as plt
from scipy.io import wavfile as wav

def wav_to_spectrogram(audio_path, save_path, spectrogram_dimensions=(64, 64), noverlap=16, cmap='gray_r'):
    sample_rate, samples = wav.read(audio_path)
    if samples.ndim > 1:
        samples = samples.mean(axis=1)  # convert to mono
    fig = plt.figure()
    fig.set_size_inches((spectrogram_dimensions[0]/fig.get_dpi(), spectrogram_dimensions[1]/fig.get_dpi()))
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    ax.specgram(samples, cmap=cmap, Fs=sample_rate, noverlap=noverlap)
    ax.xaxis.set_major_locator(plt.NullLocator())
    ax.yaxis.set_major_locator(plt.NullLocator())
    fig.savefig(save_path, bbox_inches="tight", pad_inches=0)


def dir_to_spectrogram(audio_dir, spectrogram_dir, spectrogram_dimensions=(64, 64), noverlap=16, cmap='gray_r'):
  file_names = [f for f in listdir(audio_dir) if isfile(join(audio_dir, f)) and '.wav' in f]
  for file_name in file_names:
    print(file_name)
    audio_path = audio_dir + file_name
    spectogram_path = spectrogram_dir + file_name.replace('.wav', '.png')
    wav_to_spectrogram(audio_path, spectogram_path, spectrogram_dimensions=spectrogram_dimensions, noverlap=noverlap, cmap=cmap)


#--> Create two folder recordings and spectrograms
#--> recordings folder should have .wav files
#--> spectrogram folder needs to be empty and then give the paths below

audio_folder = "/content/drive/MyDrive/Colab Notebooks/recordings/" # "recordings" folder path
spectrogram_folder = "/content/drive/MyDrive/Colab Notebooks/spectrograms/" # "spectrogram" folder path
dir_to_spectrogram(audio_folder, spectrogram_folder)

import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image
from keras.utils.np_utils import to_categorical

imagesDir = "/content/drive/MyDrive/Colab Notebooks/spectrograms/"
trainset = []
testset = []
for file in os.listdir(imagesDir):
  label = file.split('_')[0]
  sample_number = file.split('_')[2]
  img = image.load_img(imagesDir+file)
  if sample_number in ['110.png','111.png','112.png','113.png','114.png','115.png','116.png','117.png','118.png','119.png',
                       '120.png','121.png','122.png','123.png','124.png','125.png','126.png','127.png','128.png','129.png',
                       '130.png','131.png','132.png','133.png','134.png','135.png','136.png','137.png','138.png','139.png',
                       '140.png','141.png','142.png','143.png','144.png','145.png','146.png','147.png','148.png','149.png',]:
    testset.append([image.img_to_array(img), label])
  else:
    trainset.append([image.img_to_array(img), label])


# Get only images in the train list not the Labels
X_train = [item[0] for item in trainset]
# Get only Labels in the train list not the images
y_train = [item[1] for item in trainset]
# Get only images in the test list not the Labels
X_test = [item[0] for item in testset]
# Get only Labels in the test list not the images
y_test = [item[1] for item in testset]

# Convert list to numpy array in order to define input shape
X_train = np.asanyarray(X_train)
y_train = np.asanyarray(y_train)
X_test = np.asanyarray(X_test)
y_test = np.asanyarray(y_test)

# # convert to one hot representation
# y_train = to_categorical(y_train)
# y_test = to_categorical(y_test)

# Convert labels to numerical values
label_to_index = {label: i for i, label in enumerate(set(y_train))}
y_train_numeric = np.array([label_to_index[label] for label in y_train])
y_test_numeric = np.array([label_to_index[label] for label in y_test])

# Convert numerical labels to one-hot encoded vectors
num_classes = len(label_to_index)
y_train= to_categorical(y_train_numeric, num_classes=num_classes)
y_test= to_categorical(y_test_numeric, num_classes=num_classes)

#Normalize the images
X_train /= 255
X_test /= 255

from tensorflow.keras.models import Sequential
from keras import optimizers
from keras.layers import Dense, Activation, Flatten, Conv2D, MaxPooling2D, Dropout, BatchNormalization
from keras import models


data_shape = (X_train.shape[1], X_train.shape[2], X_train.shape[3])
def basic_cnn():
  model = Sequential()
  model.add(Conv2D(32, kernel_size=(2, 2), activation='relu', input_shape=data_shape))
  model.add(BatchNormalization())
  model.add(Conv2D(48, kernel_size=(2, 2), activation='relu'))
  model.add(BatchNormalization())
  model.add(Conv2D(120, kernel_size=(2, 2), activation='relu'))
  model.add(BatchNormalization())
  model.add(MaxPooling2D(pool_size=(2, 2)))
  model.add(Dropout(0.25))
  model.add(Flatten())
  model.add(Dense(128, activation='relu'))
  model.add(BatchNormalization())
  model.add(Dropout(0.25))
  model.add(Dense(64, activation='relu'))
  model.add(BatchNormalization())
  model.add(Dropout(0.4))
  model.add(Dense(10, activation='softmax'))
  model.compile(loss = 'categorical_crossentropy', optimizer='Adadelta', metrics=['accuracy'])
  return model

model0 = basic_cnn()
model0.summary()

model0.fit(X_train, y_train, batch_size = 50, validation_split=0.2, epochs = 200, verbose = 1)

model0.evaluate(X_test, y_test)


import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

y_pred = model0.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true_classes = np.argmax(y_test, axis=1)

confusion_mat = confusion_matrix(y_true_classes, y_pred_classes)

class_labels = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]

plt.figure(figsize=(8, 6))
sns.heatmap(confusion_mat, annot=True, fmt="d", cmap="Blues", xticklabels=class_labels, yticklabels=class_labels)
plt.title("Confusion Matrix")
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.show()

a = y_test_numeric

d = dict(enumerate(map(str, a)))
print(d)

X_test[6]

index = 6
print('ground Truth',np.argmax(y_test[index]))
print('Prediction' ,np.argmax(model0.predict(X_test[index].reshape(1,64,64,3))))


# !pip install pyyaml h5py  # Required to save models in HDF5 format
model0.save("/content/drive/MyDrive/Colab Notebooks/spoken_digit_recognition_.h5")
