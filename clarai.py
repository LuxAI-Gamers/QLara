import math
import sys
import random

import numpy as np
from lux.constants import Constants
from lux.game_constants import GAME_CONSTANTS
DIRECTIONS = Constants.DIRECTIONS

from model import QLModel


class Clara():


    C_ACTIONS = [
                 lambda x: x.build_worker(),
                 lambda x: x.research()
              ]

    W_ACTIONS = [
               lambda x: x.move(DIRECTIONS.NORTH),
               lambda x: x.move(DIRECTIONS.WEST),
               lambda x: x.move(DIRECTIONS.SOUTH),
               lambda x: x.move(DIRECTIONS.EAST),
               lambda x: x.build_city(),
               lambda x: x.pillage(),
              ]


    def __init__(self):

        self._lr = 0.01
        self._gamma = 0.95
        self._epsilon = 1.0
        self._epsilon_final = 0.01
        self._epsilon_decay = 0.995

        output_shape = len(self.C_ACTIONS + self.W_ACTIONS)

        self._model = QLModel(output_shape = output_shape)


    def play(self, game_state, observation):

        # GET NEW INPUTS
        x = self.get_env_state(game_state)

        # GET REWARD
        reward = self.compute_reward(game_state, observation)

        # TRAIN THE LAST REWARD
        y = self.update_q_function(x, reward)

        # GET NEW NEW ACTION
        actions = self.get_agent_action(y, game_state)

        # UPDATE PREVIOUS STATE
        self._last_state = {'x' : x,
                            'y' : y,
                            'game_state' : game_state,
                            'observation' : observation}

        return actions


    def init_last_state(self, game_state, observation):


        x = self.get_env_state(game_state)
        player = game_state.players[observation.player]
        output_shape = len(self.C_ACTIONS + self.W_ACTIONS)

        self._last_state = {'x' : x,
                            'game_state' : game_state,
                            'observation' : observation,
                            'y' : np.zeros((x.shape[0],
                                            x.shape[1],
                                            output_shape))}


    def compute_reward(self, game_state, observation):
        """
        """
        new_reward = observation['reward']
        old_reward = self._last_state['observation']['reward']

        reward = new_reward/old_reward -1 if old_reward!=0 else 0.1
        reward = reward + self._gamma * np.amax(self._last_state['y'], axis=2)

        return reward


    def update_q_function(self, x, reward):

        new_y = self._model.predict(x)

        old_x = self._last_state['x']
        old_y = self._last_state['y']
        player= self._last_state['game_state'].players[0]

        if random.random() > self._epsilon:
            new_y = np.random.rand(*new_y.shape)

        # SPLIT UNIT AND CITY PREDICTIONS
        y_unit = new_y[:,:,0:len(self.W_ACTIONS)]
        y_city = new_y[:,:,len(self.W_ACTIONS):-1]

        # CITY LOSS
        best_actions_map = np.argmax(y_city, axis=2)
        for city in player.cities.values():
            for city_tile in city.citytiles:
                i, j = city_tile.pos.x, city_tile.pos.y
                idx = best_actions_map[i,j]
                old_y[i, j][idx] = reward[i, j]

        # UNITS LOSS
        best_actions_map = np.argmax(y_unit, axis=2)
        for unit in player.units:
            i, j = unit.pos.x, unit.pos.y
            idx = best_actions_map[i, j]
            old_y[i, j][idx] = reward[i, j]

        # FIT MODEL
        old_y = (1-self._lr) * new_y + self._lr * old_y
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
        r = 2*r/r.max()-1

        # Map units
        shape = (w, h, 6)
        u = np.zeros(6*w*h).reshape(*shape)
        units = game_state.players[0].units + game_state.players[1].units
        for i in units:
            u[i.pos.y][i.pos.x] = [i.type,
                                   i.team,
                                   i.cooldown,
                                   i.cargo.wood,
                                   i.cargo.coal,
                                   i.cargo.uranium]

        # Cities in map
        e =  list(game_state.players[0].cities.values())
        e += list(game_state.players[1].cities.values())
        shape = (w, h, 4)
        c = np.zeros(4*w*h).reshape(*shape)
        for city in e:
            citytiles = city.citytiles
            for i in citytiles:
                c[i.pos.y][i.pos.x] = [i.cooldown,
                                       city.fuel,
                                       city.light_upkeep,
                                       city.team]

        return np.dstack([r, u, c])


    def get_agent_action(self, new_y, game_state):

        actions = []
        player = game_state.players[0]

        # SPLIT UNIT AND CITY PREDICTIONS
        y_unit = new_y[:,:,0:len(self.W_ACTIONS)]
        y_city = new_y[:,:,len(self.W_ACTIONS):-1]

        # GET BEST UNIT ACTION
        best_actions_map = np.argmax(y_unit, axis=2)
        for unit in player.units:
            idx = best_actions_map[unit.pos.y, unit.pos.x]
            if unit.can_act():
                actions.append(self.W_ACTIONS[idx](unit))

        # GET BEST CITY ACTION
        best_actions_map = np.argmax(y_city, axis=2)
        for city in player.cities.values():
            for city_tile in city.citytiles:
                idx = best_actions_map[city_tile.pos.y, city_tile.pos.x]
                if city_tile.can_act():
                    actions.append(self.C_ACTIONS[idx](city_tile))

        return actions
