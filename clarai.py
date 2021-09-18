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
               ('move north',   lambda x: x.move(DIRECTIONS.NORTH)),
               ('move west',    lambda x: x.move(DIRECTIONS.WEST)),
               ('move south',   lambda x: x.move(DIRECTIONS.SOUTH)),
               ('move east',    lambda x: x.move(DIRECTIONS.EAST)),
               ('build city',   lambda x: x.buid_city()),
               ('pillage',      lambda x: x.pillage()),
               ('build_worker', lambda x: x.build_worker()),
               ('research',     lambda x: x.research())
    ]

    def __init__(self, game_state, observation):

        self._lr = 0.01
        self._gamma = 0.95
        self._epsilon = 1.0
        self._epsilon_final = 0.01
        self._epsilon_decay = 0.995

        map_shape = (game_state.map.width, game_state.map.height,10)
        self._model = QLModel(input_shape = map_shape,
                              output_shape = len(self.ACTIONS))

        player = game_state.players[observation.player]
        x = self.get_env_state(game_state)

        self._last_state = {'x' : x,
                            'y' : np.zeros((map_shape[0],map_shape[1],len(self.ACTIONS))),
                            'player' : player,
                            'observation' : observation}

    def play(self,game_state, observation):

        # GET PLAYER
        player = game_state.players[observation.player]

        # GET REWARD
        reward = self.compute_reward(game_state, observation)

        # GET NEW INPUTS
        x = self.get_env_state(game_state)

        # TRAIN THE LAST REWARD
        y = self.update_q_function(x, reward)

        # GET NEW NEW ACTION
        actions = self.get_agent_actions(y, player)

        # UPDATE PREVIOUS STATE
        self._last_state = {'x' : x,
                            'y' : y,
                            'player' : player,
                            'observation' : observation}

        return actions


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

        if random.random() < self._epsilon:
            new_y = np.random.rand(*new_y.shape)

        option = np.argmax(y[:,:,0:6], axis=2)
        for unit in self._last_state['player'].units:
            x, y = unit.pos.x, unit.pos.y
            o = option[x, y]
            old_y[x, y][o] = reward[x, y]

        old_y = new_y + self._lr * old_y
        self._model.fit(old_x, old_y)

        return new_y


    def get_env_state(self, game_state):

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


    def get_agent_action(y, player):

        actions = []

        option = np.argmax(y[:,:,0:6], axis=2)
        for unit in player.units:
            idx = option[unit.pos.y, unit.pos.x]
            actions.append(self.ACTIONS[idx](unit))

        option = np.argmax(y[:,:,6:], axis=2)
        for city in player.cities.values():
            for city_tile in city.citytiles:
                idx = option[city_tile.pos.y, city_tile.pos.x]
                actions.append(self.ACTIONS[idx](unit))

        return actions
