import math
import sys
import random

import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from tensorflow import keras
from keras import layers

from lux.game import Game
from lux.game_map import Cell, RESOURCE_TYPES, Position
from lux.game_objects import Unit
from lux.constants import Constants
from lux.game_constants import GAME_CONSTANTS
from lux import annotate


DIRECTIONS = Constants.DIRECTIONS
game_state = None
last_state = {}
learning_rate = 0.01
gamma = 0.95
epsilon = 1.0
epsilon_final = 0.01
epsilon_decay = 0.995
rl_model = None
last_reward = 0


def get_inputs_model(game_state):
    
    # Map shape
    w, h = game_state.map.width, game_state.map.height
    
    # Map resources
    r = [
        [0 if game_state.map.map[i][j].resource == None
         else game_state.map.map[i][j].resource.amount 
         for i in range(w)] for j in range(h)
    ]

    r = np.array(r).reshape(h, w, 1)

    # Map units
    shape = (w, h, 5)
    u = np.zeros(5*w*h).reshape(*shape)
    units = game_state.players[0].units
    for i in units:
        u[i.pos.y][i.pos.x] = [i.type, 
                               i.cooldown,
                               i.cargo.wood,
                               i.cargo.coal,
                               i.cargo.uranium]

    # Cities in map
    e = game_state.players[1].cities
    shape = (w, h, 4)
    c = np.zeros(4*w*h).reshape(*shape)
    for k in e:
        citytiles = e[k].citytiles
        for i in citytiles:
            c[i.pos.y][i.pos.x] = [i.cooldown,
                                   e[k].fuel,
                                   e[k].light_upkeep,
                                   e[k].team]

    return np.dstack([r, u, c])


def model(game_state):
    inputs = keras.Input(shape = get_inputs_model(game_state).shape, 
    name = 'Game map')

    m = layers.Conv2D(8, (1, 1), activation='relu')(inputs)
    m = layers.Conv2D(8, (1, 1), activation='relu')(m)
    m = layers.Conv2D(8, (1, 1), activation='relu')(m)
    output = layers.Dense(8, activation='softmax', name='direction')(m)

    model = keras.Model(inputs=inputs, outputs=output)
    model.compile(loss='mse', optimizer='adam')

    print(model.summary())

    return model


def predict_action(y, player):
    option = np.argmax(y, axis=2)

    actions = []

    for i in player.units:
        o = option[i.pos.y, i.pos.x]
        d = 'csnwe#############'[o]

        if o < 5: actions.append(i.move(d))
        elif o == 5 and i.can_build(game_state.map): actions.append(i.build_city())

    city_tiles = []
    for city in player.cities.values():
        for city_tile in city.citytiles:
            if option[city_tile.pos.y, city_tile.pos.x]==6:
                actions.append(city_tile.research())
            elif option[city_tile.pos.y, city_tile.pos.x]==7:
                actions.append(city_tile.build_worker())

    return actions, option
    

def agent(observation, configuration):
    global game_state, epsilon, rl_model, last_reward

    ### Do not edit ###
    if observation["step"] == 0:
        game_state = Game()
        game_state._initialize(observation["updates"])
        game_state._update(observation["updates"][2:])
        game_state.id = observation.player
        if not rl_model: 
            rl_model = model(game_state)
    else:
        game_state._update(observation["updates"])
    
    actions = []

    ### AI Code goes down here! ### 
    player = game_state.players[observation.player]

    x = get_inputs_model(game_state)
    y = rl_model.predict(np.asarray([x]))[0]

    if random.random() < epsilon: y = np.random.rand(*y.shape)

    print('epsilon ', epsilon, end=' | ')
    actions, option = predict_action(y, player)
    print(actions)
    print('reward ', observation['reward'])

    if observation.player in last_state:
        _x, _y, _player, _option = last_state[observation.player]
        state, _, reward = _x, x, observation['reward']

        if reward > last_reward: r = 1
        elif reward < last_reward: r = -0.1
        else: r = 0.1

        r = r + gamma * np.amax(_y, axis=2)

        for i in _player.units:
            _y[i.pos.y, i.pos.x][_option[i.pos.y, i.pos.x]] = r[i.pos.x, i.pos.y]
        
        _y = y + learning_rate * _y

        rl_model.fit(np.asarray([state]), np.asarray([_y]), epochs=1, verbose=1)

        if epsilon > epsilon_final: epsilon *= epsilon_decay
    
    last_state[observation.player] = [x, y, player, option]
    last_reward = observation['reward']
    return actions

if __name__=='__main__':

    import json
    from kaggle_environments import make
    from IPython.display import clear_output 

    episodes = 10
    for ep in range(episodes):
        clear_output()
        print(f"=== Episode {ep} ===")
        env = make("lux_ai_2021",
                configuration={"seed": 562124210,
                                "loglevel": 0,
                                "annotations": True},
                debug=True)

        steps = env.run([agent, "simple_agent"])
    print([step[0]['action'] for step in steps])

    replay = env.toJSON()
    with open("replay.json", "w") as f:
        json.dump(replay, f)
