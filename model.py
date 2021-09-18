import math
import sys
import random

import numpy as np
import tensorflow as tf
import tensorflow_hub as hub



class QLModel():

    def __init__(self, input_shape, output_shape):

        self._input_shape = input_shape
        self._output_shape = output_shape
        self.build()

    def build(self):

        inputs = tf.keras.Input(shape = self._input_shape,
                                name = 'Game map')

        x = tf.keras.layers.Conv2D(self._output_shape,
                                   (1, 1),
                                   activation='relu')(inputs)

        x = tf.keras.layers.Conv2D(self._output_shape,
                                   (1, 1),
                                   activation='relu')(x)

        x = tf.keras.layers.Conv2D(self._output_shape,
                                   (1, 1),
                                   activation='relu')(x)

        x = tf.keras.layers.Dense(self._output_shape,
                                  activation='softmax',
                                  name='direction')(x)

        model = tf.keras.Model(inputs=inputs, outputs=x)

        model.compile(loss='mse', optimizer='adam')
        print(model.summary())
        self._model=model


    def fit(self, env_state, reward):

        self._model.fit(np.asarray([env_state]),
                        np.asarray([reward]),
                        epochs=1,
                        verbose=1)


    def predict(self, env_state):
 
        y = self._model.predict(np.asarray([env_state]))[0]


    def save(self, model_path):
        self._model.save(model_path)


    def load(self, model_path):
        self._model = tf.keras.model.load_model(model_path)
