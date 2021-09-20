import math
import sys
import random

import numpy as np
import tensorflow as tf
import tensorflow_hub as hub



class QLModel():

    def __init__(self, output_shape):

        self._output_shape = output_shape
        self._models = {}
        for dim in [32,24,16,12]:
            inputs = tf.keras.Input(shape=(dim,dim,10),
                                    name='Game map')

            self._models[dim] = self.build_network(inputs)


    def build_network(self, inputs):

        x = tf.keras.layers.Conv2D(self._output_shape,
                                   (3, 3),
                                   padding='same',
                                   activation='relu')(inputs)

        x = tf.keras.layers.Conv2D(self._output_shape,
                                   (3, 3),
                                   padding='same',
                                   activation='relu')(x)

        x = tf.keras.layers.Conv2D(self._output_shape,
                                   (3, 3),
                                   padding='same',
                                   activation='relu')(x)

        x = tf.keras.layers.Dense(self._output_shape,
                                  activation='softmax',
                                  name='direction')(x)

        model = tf.keras.Model(inputs=inputs, outputs=x)

        model.compile(loss='mse', optimizer='adam')

        return model


    def fit(self, env_state, reward):

        dim = env_state.shape[0]
        self._models[dim].fit(np.asarray([env_state]),
                              np.asarray([reward]),
                              epochs=1,
                              verbose=1)


    def predict(self, env_state):
        dim = env_state.shape[0]
        return self._models[dim].predict(np.asarray([env_state]))[0]


    def save(self, model_path):
        self._model.save(model_path)


    def load(self, model_path):
        self._model = tf.keras.model.load_model(model_path)
