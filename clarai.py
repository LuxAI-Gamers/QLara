import math
import sys
import random

import numpy as np
from lux.constants import Constants
from lux.game_constants import GAME_CONSTANTS
DIRECTIONS = Constants.DIRECTIONS

from model import QLModel

class Clara():


    ACTIONS = [
               ('worker', lambda x: x.move(DIRECTIONS.NORTH)),
               ('worker', lambda x: x.move(DIRECTIONS.WEST)),
               ('worker', lambda x: x.move(DIRECTIONS.SOUTH)),
               ('worker', lambda x: x.move(DIRECTIONS.EAST)),
               ('worker', lambda x: x.build_city()),
               ('worker', lambda x: x.pillage()),
               ('city',   lambda x: x.build_worker()),
               ('city',   lambda x: x.research())
    ]

    def __init__(self):

        self._lr = 0.01
        self._gamma = 0.95
        self._epsilon = 1.0
        self._epsilon_final = 0.01
        self._epsilon_decay = 0.995

        self._model = QLModel(input_shape = 10,
                              output_shape = len(self.ACTIONS))





    def play(self, game_state, observation):

        # GET PLAYER
        player = game_state.players[observation.player]

        # GET NEW INPUTS
        x = self.get_env_state(game_state)

        # GET REWARD
        reward = self.compute_reward(game_state, observation)

        # TRAIN THE LAST REWARD
        y = self.update_q_function(x, reward)

        # GET NEW NEW ACTION
        actions = self.get_agent_action(y, player)

        # UPDATE PREVIOUS STATE
        self._last_state = {'x' : x,
                            'y' : y,
                            'player' : player,
                            'observation' : observation}

        return actions


    def init_last_state(self, game_state, observation):


        x = self.get_env_state(game_state)
        player = game_state.players[observation.player]

        self._last_state = {'x' : x,
                            'player' : player,
                            'observation' : observation,
                            'y' : np.zeros((x.shape[0],
                                            x.shape[1],
                                            len(self.ACTIONS)))}


    def compute_reward(self, game_state, observation):
        """
        """
        new_reward = observation['reward']
        old_reward = self._last_state['observation']['reward']

        reward = 1.0 if new_reward>old_reward else -0.1
        reward = reward + self._gamma * np.amax(self._last_state['y'], axis=2)

        return reward


    def update_q_function(self, x, reward):

        new_y = self._model.predict(x)

        old_x = self._last_state['x']
        old_y = self._last_state['y']

        if random.random() > self._epsilon:
            new_y = np.random.rand(*new_y.shape)

#        option = np.argmax(new_y[:,:,6:], axis=2)
#        for city in self._last_state['player'].cities.values():
#            for city_tile in city.citytiles:
#                i, j = city_tile.pos.x, city_tile.pos.y
#                idx = option[i,j]+6
#                old_y[i, j][idx] = reward[i, j]

        option = np.argmax(new_y[:,:,0:6], axis=2)
        for unit in self._last_state['player'].units:
            i, j = unit.pos.x, unit.pos.y
            idx = option[i, j]
            old_y[i, j][idx] = reward[i, j]

        old_y = new_y + self._lr * old_y
        self._model.fit(old_x, old_y)

        return new_y


    def get_env_state(self, game_state):

        # Map shape
        w, h = game_state.map.width, game_state.map.height
        # Map resources
        r = [
            [0 if game_state.map.map[i][j].resource is None
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


    def get_agent_action(self, y, player):

        actions = []

        best_options_map = np.argmax(y[:,:,0:6], axis=2)
        options = [act for act in self.ACTIONS if act[0]=='worker']
        for unit in player.units:
            idx = best_options_map[unit.pos.y, unit.pos.x]
            actions.append(options[idx][1](unit))

        best_options_map = np.argmax(y[:,:,6:], axis=2)
        options = [act for act in self.ACTIONS if act[0]=='city']
        for city in player.cities.values():
            for city_tile in city.citytiles:
                idx = best_options_map[city_tile.pos.y, city_tile.pos.x]
                actions.append(options[idx][1](city_tile))

        return actions
