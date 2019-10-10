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


ROOT_PATH = "/Users/Realm/Desktop/"
train_data_directory = os.path.join(ROOT_PATH, "TrafficSigns/Training")
test_data_directory = os.path.join(ROOT_PATH, "TrafficSigns/Testing")
# PracticeData = os.path.join(ROOT_PATH, "TrafficSigns/sort")
# practiceImage,practiceLabel = load_data(PracticeData)

train_images, train_labels = load_data(train_data_directory)
test_images,test_labels = load_data(test_data_directory)
#
practiceImage = train_images
train_images28 = [transform.resize(image, (28, 28)) for image in train_images]
test_images28 = [transform.resize(image,(28,28)) for image in test_images]

train_images = np.array(train_images28)
test_images = np.array(test_images28)
#
train_images = rgb2gray(train_images)
test_images = rgb2gray(test_images)

train_images = train_images[:1000].reshape(-1,28*28)/255.0
test_images = test_images[:1000].reshape(-1,28*28)/255.0

new_model = keras.models.load_model('model.h5')
new_model.summary()
loss, acc = new_model.evaluate(test_images, test_labels)
print("Untrained model, accuracy: {:5.2f}%".format(100*acc))
# # plt.imshow(practiceTest[0])
# # plt.show()
img = (np.expand_dims(train_images[0],0))
print(img.shape)
#
pridiction = new_model.predict(img)
print(pridiction)
print(np.argmax(pridiction[0]))

plt.imshow(practiceImage[0])
plt.show()

