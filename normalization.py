from keras.layers import Layer

class Normalization(Layer):
   def call(self, x):
     return x / 255.0 - 0.5
