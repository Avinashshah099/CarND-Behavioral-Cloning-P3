import csv
import cv2
import numpy as np
from keras.models import Sequential
from keras.layers import Flatten, Dense, Conv2D, MaxPooling2D, Cropping2D
from keras.callbacks import ModelCheckpoint
from keras.utils import plot_model
from sklearn.model_selection import train_test_split
import sklearn
from math import ceil
from random import shuffle
from normalization import Normalization
import os

class Model:


    """
    
        Model class
    
    """

    def __init__(self):
        

        """
        
            Model constructor
            
        """
        
        filenames = []
        for root, dirs, files in os.walk("./data"):
            for file in files:
                if file.endswith("driving_log.csv"):
                    filenames.append(os.path.join(root, file))
                    
        samples = []
        for filename in filenames:
            with open(filename) as csvfile:
                reader = csv.reader(csvfile)
                for sample in reader:
                    samples.append(sample)
        
        # Split images to test set and validation set.
        self.train_samples, self.validation_samples = train_test_split(samples, test_size=0.2)
        train = len(self.train_samples)
        validation = len(self.validation_samples)
        total = train + validation
        print('Train set size: {} ({}%)'.format(train, round(train*100/total)))
        print('Validation set size: {} ({}%)'.format(validation, round(validation*100/total)))
        print('Total size: {} (100%)'.format(total))
        print('Total train images number: {} ({}%)'.format(6*train, round(train*100/total)))
        print('Total validation images nimber: {} ({}%)'.format(6*validation, round(validation*100/total)))
        print('Total images number: {} (100%)'.format(6*total))



    def generator(self, samples, batch_size=16):


        """
            Images batch generator that delivers batch_size mutiplied by six images. Each batch consist of images from center, left and right camera and its
            horizontally flipped representation.
            
            samples    - array with paths to images and steering angles,
            batch_size - size of batch of images to be returned.
            
            
        """
        
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
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    angle = float(batch_sample[3])
                    images.append(image)
                    angles.append(angle)
                    images.append(cv2.flip(image, 1))
                    angles.append(-angle)

                    image = cv2.imread(batch_sample[1])
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    angle = float(batch_sample[3]) + correction
                    images.append(image)
                    angles.append(angle)
                    images.append(cv2.flip(image, 1))
                    angles.append(-angle)

                    image = cv2.imread(batch_sample[2])
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    angle = float(batch_sample[3]) - correction
                    images.append(image)
                    angles.append(angle)
                    images.append(cv2.flip(image, 1))
                    angles.append(-angle)

                X_train = np.array(images)
                y_train = np.array(angles)
                yield sklearn.utils.shuffle(X_train, y_train)


    def train(self, filename='model.h5', batch_size=16):
        

        """
        
            Train the model and store results in specified file.
            
            filename - name of the file. 
            
        """
        
        # Create model
        model = Sequential()
        
        # Preprocess the images
        model.add(Cropping2D(cropping=((55,25), (0, 0)), input_shape=(160, 320, 3))) # trimming
        model.add(Normalization())                                                   # normalization

        # Create network architecture based on https://devblogs.nvidia.com/deep-learning-self-driving-cars/
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
        model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
        plot_model(model, to_file='images/model.png', show_shapes=True)
        
        # train the model
        history = model.fit_generator(self.generator(self.train_samples, batch_size=batch_size), steps_per_epoch=ceil(len(self.train_samples)/batch_size),
        validation_data=self.generator(self.validation_samples, batch_size=batch_size), validation_steps=ceil(len(self.validation_samples)/batch_size),
        epochs=5, verbose=1, callbacks=[ModelCheckpoint(filename, verbose=1, save_best_only=True)])
        print(history.history.keys())
        print('Loss:')
        print(history.history['loss'])
        print('Validation Loss:')
        print(history.history['val_loss'])
        print('Accuracy:')
        print(history.history['acc'])
        

Model().train()
