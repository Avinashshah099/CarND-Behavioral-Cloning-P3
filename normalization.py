from keras.layers import Layer
import numpy as np
import tensorflow as tf

class Normalization(Layer):
    def call(self, x):
        mean, var = tf.nn.moments(x, axes=[0])
        return (x - mean)/var
