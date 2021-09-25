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
        self._epsilon = 0.95
        self._epsilon_final = 0.01
        self._epsilon_decay = 0.995

        output_shape = len(self.C_ACTIONS + self.W_ACTIONS)

        self._model = QLModel(output_shape = output_shape)


    def play(self, game_state, observation):

        self._new_state =  {'x' : None,
                            'y' : None,
                            'game_state' : game_state,
                            'observation' : observation}

        # GET NEW INPUTS
        self._new_state['x'] = self.get_env_state()

        # GET REWARD
        reward = self.compute_reward()

        # TRAIN THE LAST REWARD
        self._new_state['y'] = self.update_q_function(reward)

        # GET NEW NEW ACTION
        actions = self.get_agent_action()

        # UPDATE PREVIOUS STATE
        self._old_state = self._new_state
        self._old_state['actions'] = actions

        print(actions)

        return actions


    def init_old_state(self, game_state, observation):

        self._new_state = {'x' : None,
                           'y' : None,
                           'actions' : [],
                           'game_state' : game_state,
                           'observation' : observation}

        output_shape = len(self.C_ACTIONS + self.W_ACTIONS)
        x = self.get_env_state()
        y = np.zeros((x.shape[0],x.shape[1], output_shape))

        self._new_state['x'] = x
        self._new_state['y'] = y
        self._old_state = self._new_state


    def compute_reward(self):
        """
        """
        old_y = self._old_state['y']

        new_state = self._new_state['game_state']
        old_state = self._old_state['game_state']

        new_reward = self._new_state['observation']['reward']
        old_reward = self._old_state['observation']['reward']

        reward = new_reward/old_reward -1 if old_reward!=0 else 0.1
        reward = reward + self._gamma * np.amax(old_y, axis=2)

        reward = self.validate(reward)

        return reward


    def validate(self, reward):

        actions    = self._old_state['actions']
        old_game_state = self._old_state['game_state']
        old_observation = self._old_state['observation']
        old_player = old_game_state.players[old_observation.player]

        new_game_state = self._new_state['game_state']
        new_observation = self._new_state['observation']
        new_player = new_game_state.players[new_observation.player]

        units_that_acted = [action.split(' ')[1] for action in actions]

        for new_unit in new_player.units:
            for old_unit in old_player.units:
                if  new_unit.id            ==  old_unit.id         and \
                    new_unit.pos.x         ==  old_unit.pos.x      and \
                    new_unit.pos.y         ==  old_unit.pos.y      and \
                    new_unit.cargo.wood    ==  old_unit.cargo.wood and \
                    new_unit.cargo.coal    ==  old_unit.cargo.coal and \
                    new_unit.cargo.uranium ==  old_unit.cargo.uranium:

                    if new_unit.id in units_that_acted:
                        reward[new_unit.pos.y,new_unit.pos.x] = -2

        return reward


    def update_q_function(self,reward):

        new_x = self._new_state['x']
        new_y = self._model.predict(new_x)

        old_x = self._old_state['x']
        old_y = self._old_state['y']

        game_state = self._old_state['game_state']
        observation = self._old_state['observation']

        player = game_state.players[observation.player]

        if random.random() > self._epsilon:
            new_y = np.random.rand(*new_y.shape)

        # SPLIT UNIT AND CITY PREDICTIONS
        y_unit = new_y[:,:,0:len(self.W_ACTIONS)]
        y_city = new_y[:,:,len(self.W_ACTIONS):-1]

        # CITY LOSS
        best_actions_map = np.argmax(y_city, axis=2)
        for city in player.cities.values():
            for city_tile in city.citytiles:
                j,i = city_tile.pos.x, city_tile.pos.y
                idx = best_actions_map[i,j]
                old_y[i,j][idx] = reward[i,j]

        # UNITS LOSS
        best_actions_map = np.argmax(y_unit, axis=2)
        for unit in player.units:
            j,i = unit.pos.x, unit.pos.y
            idx = best_actions_map[i,j]
            old_y[i,j][idx] = reward[i,j]

        # FIT MODEL
        old_y = (1-self._lr) * new_y + self._lr * old_y
        self._model.fit(old_x, old_y)

        return new_y


    def get_env_state(self):


        game_state = self._new_state['game_state']
        observation = self._new_state['observation']
        player = game_state.players[observation.player]
        opponent = game_state.players[(observation.player+1)%2]

        # MAP SHAPE
        w, h = game_state.map.width, game_state.map.height

        # MAP RESOURCES
        r = [
            [0 if game_state.map.map[i][j].resource is None
             else game_state.map.map[i][j].resource.amount
             for i in range(w)] for j in range(h)
        ]

        r = np.array(r).reshape(h, w, 1)
        r = 2*r/r.max()-1

        # MAP UNITS
        shape = (w, h, 6)
        u = np.zeros(6*w*h).reshape(*shape)
        units = player.units + opponent.units
        for i in units:
            u[i.pos.y][i.pos.x] = [i.type,
                                   i.team,
                                   i.cooldown,
                                   i.cargo.wood,
                                   i.cargo.coal,
                                   i.cargo.uranium]

        # CITIES IN MAP
        e  = list(player.cities.values())
        e += list(opponent.cities.values())

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


    def get_agent_action(self):

        actions = []
        new_y = self._new_state['y']

        game_state = self._new_state['game_state']
        observation = self._new_state['observation']
        player = game_state.players[observation.player]

        # SPLIT UNIT AND CITY PREDICTIONS
        y_unit = new_y[:,:,0:len(self.W_ACTIONS)]
        y_city = new_y[:,:,len(self.W_ACTIONS):]

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

