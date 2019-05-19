from keras.layers import Layer
import numpy as np
import tensorflow as tf

class Normalization(Layer):
    def call(self, x):
        return x/127.5 - 1.0
