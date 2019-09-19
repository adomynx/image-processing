import os
import skimage
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow import keras
from skimage.color import rgb2gray
from skimage import transform
from tensorflow.python.saved_model import tag_constants

def load_data(data_directory):
    directories = [d for d in os.listdir(data_directory)
                   if os.path.isdir(os.path.join(data_directory, d))]
    labels = []
    images = []
    for d in directories:
        label_directory = os.path.join(data_directory, d)
        file_names = [os.path.join(label_directory, f)
                      for f in os.listdir(label_directory)
                      if f.endswith(".ppm")]
        for f in file_names:
            images.append(skimage.data.imread(f))
            labels.append(int(d))
    return images, labels

ROOT_PATH="/Users/Realm/Desktop"

train_data_directory = os.path.join(ROOT_PATH, "TrafficSigns/Training")
test_data_directory = os.path.join(ROOT_PATH, "TrafficSigns/Testing")

train_images, train_labels = load_data(train_data_directory)
test_images,test_labels = load_data(test_data_directory)

train_images28 = [transform.resize(image, (28, 28)) for image in train_images]
test_images28 = [transform.resize(image,(28,28)) for image in test_images]

train_images = np.array(train_images28)
test_images = np.array(test_images28)

train_images = rgb2gray(train_images)
test_images = rgb2gray(test_images)

train_images = train_images[:1000].reshape(-1,28*28)/255.0
test_images = test_images[:1000].reshape(-1,28*28)/255.0

model = keras.Sequential([
    keras.layers.Flatten(input_shape=(784,)),
    keras.layers.Dense(128, activation=tf.nn.relu),
    keras.layers.Dense(21, activation=tf.nn.softmax)
])
model.compile(optimizer=tf.keras.optimizers.Adam(),
                loss=tf.keras.losses.sparse_categorical_crossentropy,
                metrics=['accuracy'])

model.summary()
model.fit(train_images, train_labels, epochs=100, batch_size=5, verbose=1);
loss, acc = model.evaluate(test_images, test_labels)
print("Untrained model, accuracy: {:5.2f}%".format(100*acc))
model.save("model.h5")