import csv
import cv2
import numpy as np
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Conv2D, MaxPooling2D, Cropping2D, Dropout
from sklearn.model_selection import train_test_split
import sklearn
from math import ceil
from random import shuffle
import matplotlib.pyplot as plt
from normalization import Normalization

samples = []
for i in range(1,3):
    with open('./data/track'+str(i)+'/driving_log.csv') as csvfile:
        reader = csv.reader(csvfile)
        for sample in reader:
            samples.append(sample)

with open('./data/track1/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for sample in reader:
        samples.append(sample)

train_samples, validation_samples = train_test_split(samples, test_size=0.2)

def generator(samples, batch_size=32):
    num_samples = len(samples)
    correction=0.2
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                image = cv2.imread(batch_sample[0])
                angle = float(batch_sample[3])
                images.append(image)
                angles.append(angle)
                images.append(cv2.flip(image, 1))
                angles.append(-angle)

                image = cv2.imread(batch_sample[1])
                angle = float(batch_sample[3]) + correction
                images.append(image)
                angles.append(angle)
                images.append(cv2.flip(image, 1))
                angles.append(-angle)

                image = cv2.imread(batch_sample[2])
                angle = float(batch_sample[3]) - correction
                images.append(image)
                angles.append(angle)
                images.append(cv2.flip(image, 1))
                angles.append(-angle)

            # trim image to only see section with road
            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)

# Set our batch size
batch_size=16

# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=batch_size)
validation_generator = generator(validation_samples, batch_size=batch_size)

#ch, row, col = 3, 80, 320  # Trimmed image format

model = Sequential()
model.add(Cropping2D(cropping=((55,25), (0, 0)), input_shape=(160, 320, 3)))
#model.add(Lambda(lambda x: normalization, input_shape=(80, 320, 3)))
model.add(Normalization())

model.add(Conv2D(24, (5, 5), strides=(2,2), activation='relu'))
model.add(Conv2D(36, (5, 5), strides=(2,2), activation='relu'))
model.add(Conv2D(48, (5, 5), strides=(2,2), activation='relu'))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))
model.compile(loss='mse', optimizer='adam')
model.fit_generator(train_generator, steps_per_epoch=ceil(len(train_samples)/batch_size), validation_data=validation_generator, validation_steps=ceil(len(validation_samples)/batch_size), epochs=2, verbose=1)
model.save('model.h5')

