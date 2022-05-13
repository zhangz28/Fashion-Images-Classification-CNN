import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow import keras
from keras.datasets import fashion_mnist
from keras.datasets import cifar10

batch_size = 128
num_classes = 10
epochs = 1

(x_train, y_train), (x_test, y_test) = keras.datasets.fashion_mnist.load_data()
x_train.shape
plt.imshow(x_train[0,:,:])
x_train.dtype
y_train.shape
y_train[0]
np.unique(y_train)

x_train = x_train.astype("float32") / 255
x_test = x_test.astype("float32") / 255

x_train = np.expand_dims(x_train, -1)
x_test = np.expand_dims(x_test, -1)
print("x_train shape:", x_train.shape)
print(x_train.shape[0], "train samples")
print(x_test.shape[0], "test samples")

### convert to one-hot encoding
y_train_val = keras.utils.to_categorical(y_train, num_classes)
y_test_val = keras.utils.to_categorical(y_test, num_classes)

y_train_val.shape

model3 = keras.models.Sequential()
model3.add(keras.Input(shape=(28, 28, 1, ))) # The input image size would be 28*28, and the input data shape would be (batch_size, 28, 28, 1)
model3.add(keras.layers.Conv2D(filters=32, kernel_size=3, activation="relu"))
model3.add(keras.layers.Conv2D(filters=64, kernel_size=3, activation="relu"))
model3.add(keras.layers.Flatten())
model3.add(keras.layers.Dense(64, activation="relu"))
model3.add(keras.layers.Dense(10, activation="softmax"))

print(model3.summary())

# example: configure the CNN defined in model3

model3.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-3),
               loss="categorical_crossentropy",
               metrics=["accuracy"])

model3.fit(x_train, y_train_val,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_split=0.2)

score = model3.evaluate(x_test, y_test_val, verbose=0)

print("Test loss: {0}".format(score[0]))
print("Test accuracy: {0}".format(score[1]))
