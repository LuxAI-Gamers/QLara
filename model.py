import os
import time
from functools import partial

import numpy as np
import tensorflow as tf
#import tensorflow_hub as hub



class QLModel():

    def __init__(self, output_shape):

        self._output_shape = output_shape
        self._models = {}
        for dim in [32,24,16,12]:
            inputs = tf.keras.Input(shape=(dim,dim,11),
                                    name='Game map')

            self._models[dim] = self.build_network(inputs)


    def build_network(self, inputs):

        # Inception
        x = self._inception_block(self._output_shape)(inputs)
        x = self._inception_block(self._output_shape)(x)
        x = self._inception_block(self._output_shape)(x)
        x = self._inception_block(self._output_shape)(x)
        x = self._inception_block(self._output_shape)(x)

        # Residual connection
        x = tf.keras.layers.concatenate([x, inputs], axis=-1)

        x = tf.keras.layers.Dense(self._output_shape,
                                  activation='softmax',
                                  name='direction')(x)

        model = tf.keras.Model(inputs=inputs, outputs=x)

        model.compile(loss=tf.keras.losses.CategoricalCrossentropy(),
                      optimizer='adam')

        return model


    def _inception_block(self, filters):

        f1 = (filters-1)//3
        f2 = (filters-1)//3
        f3 = (filters-1)//3
        f4 = 1 + (filters-1) % 3

        def inception_function(layer_in, f1, f2, f3, f4):
            # 1x1 conv
            conv1 = tf.keras.layers.Conv2D(f1,
                                         (1,1),
                                         padding='same',
                                         activation='relu')(layer_in)
            # 3x3 conv
            conv3 = tf.keras.layers.Conv2D(f2,
                                           (3,3),
                                           padding='same',
                                           activation='relu')(layer_in)
            # 5x5 conv
            conv5 = tf.keras.layers.Conv2D(f3,
                                           (5,5),
                                           padding='same',
                                           activation='relu')(layer_in)
            # 3x3 max pooling
            pool_fun = tf.keras.layers.MaxPooling2D((3,3),
                                                    strides=(1,1),
                                                     padding='same')

            pools = [pool_fun(layer_in) for i in range(f4)]

            # concatenate filters, assumes filters/channels last
            layers = [conv1, conv3, conv5] + pools
            layer_out = tf.keras.layers.concatenate(layers, axis=-1)

            return layer_out

        return partial(inception_function,f1=f1,f2=f2,f3=f3,f4=f4)


    def fit(self, env_state, reward):

        dim = env_state.shape[0]
        self._models[dim].fit(np.asarray([env_state]),
                              np.asarray([reward]),
                              epochs=1,
                              verbose=1)


    def predict(self, env_state):
        dim = env_state.shape[0]
        return self._models[dim].predict(np.asarray([env_state]))[0]


    def save(self, model_dir):
        timestamp = str(int(time.time()))

        os.mkdir(model_dir+'/'+timestamp)
        for name, model in self._models.items():
            model.save(model_dir+'/'+timestamp+'/'+f'model_{name}.h5')


    def load(self, model_dir):

        self._models = {}
        for file_name in os.listdir(model_dir):
            if file_name.endswith('.h5'):
                dim = int(file_name.split('.')[0][-2:])
                model = tf.keras.models.load_model(model_dir+'/'+file_name)
                self._models[dim] = model
