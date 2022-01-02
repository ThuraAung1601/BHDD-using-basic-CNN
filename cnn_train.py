import pickle
import numpy as np
import urllib.request
import gzip
import pickle
import os
from PIL import Image
from tensorflow import keras
from tensorflow.keras import layers
from keras.models import load_model

# url_path : Path of dataset in your drive
url_path = "data.pkl" 

dataset = []

with open(url_path,"rb") as file:
    dataset = pickle.load(file)
    
trainDataset = dataset["trainDataset"]
testDataset = dataset["testDataset"]

print("Train Size: {} Test Size: {}".format(len(trainDataset), len(testDataset)))

numberOfClass = 10

trainX = [list(td["image"] / 255) for td in trainDataset]
trainY = np.eye(numberOfClass)[np.array([td["label"] for td in trainDataset]).reshape(-1)].tolist() #One-hot encode training labels
testX = [list(td["image"] / 255) for td in testDataset]
testY = np.eye(numberOfClass)[np.array([td["label"] for td in testDataset]).reshape(-1)].tolist() #One-hot encode test labels

print("TrainImageShape:" + str(np.shape(trainX)),
     "TrainLabelShape:" + str(np.shape(trainY)),
     "TestImageShape:" + str(np.shape(testX)),
     "TestLabelShape:" + str(np.shape(testY)))
     
num_classes = 10
input_shape = (28, 28, 1)
model = keras.Sequential(
    [
        keras.Input(shape=input_shape),
        layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Flatten(),
        layers.Dropout(0.2),
        layers.Dense(num_classes, activation="softmax"),
    ]
)

model.summary()

# Scale images to the [0, 1] range
trainX = np.array(trainX)
testX = np.array(testX)
trainX = trainX.astype("float32") / 255
testX = testX.astype("float32") / 255

# Make sure images have shape (28, 28, 1)
trainX = np.expand_dims(trainX, -1)
testX = np.expand_dims(testX, -1)

print("x_train shape:", trainX.shape)
print(trainX.shape[0], "train samples")
print(testX.shape[0], "test samples")

trainY = np.array(trainY)
testY = np.array(testY)

batch_size = 128
epochs = 20

model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy",keras.metrics.Precision(),keras.metrics.Recall()])

history = model.fit(trainX, trainY, batch_size=batch_size, epochs=epochs, validation_split=0.1)

print(history.history.keys())

import matplotlib.pyplot as plt
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title("model accuracy")
plt.ylabel("accuracy")
plt.xlabel("epoch")
plt.legend(['train','test'],loc='lower right')
plt.show()

import matplotlib.pyplot as plt
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title("model loss")
plt.ylabel("loss")
plt.xlabel("epoch")
plt.legend(['train','test'],loc='upper right')
plt.show()

score = model.evaluate(testX, testY, verbose=0)
print("Test loss:", score[0])
print("Test accuracy:", score[1])

loss, accuracy, precision, recall = model.evaluate(testX, testY, verbose=0)
print('Loss', loss)
print('Accuracy', accuracy)
print('Precision', precision)
print('Recall', recall)

f1_score = (2* recall*precision)/(recall+precision)
print("f1_score", f1_score)

model.save('cnn.h5')  # creates a HDF5 file 'my_model.h5' 
