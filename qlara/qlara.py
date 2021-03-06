import random

import numpy as np
from lux.constants import Constants
from lux.game_constants import GAME_CONSTANTS

from .model import QLModel
from .reward import BatchReward
from .data_augmentor import DataAugmentor


DIRECTIONS = Constants.DIRECTIONS


class QLara():

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

    def __init__(self, lr=0.01, gamma=0.95, epsilon=0.95, epsilon_final=0.01,
                 epsilon_decay=0.995, batch_length=12, epochs=1):

        self._lr = lr
        self._gamma = gamma
        self._epsilon = epsilon
        self._epsilon_final = epsilon_final
        self._epsilon_decay = epsilon_decay
        self._batch_length = batch_length
        self._epochs = epochs

        output_shape = len(self.C_ACTIONS + self.W_ACTIONS)

        self._model = QLModel(output_shape=output_shape)
        self._data_augmentor = DataAugmentor()
        self._reward = BatchReward(self._lr,
                                   self._gamma,
                                   self.W_ACTIONS,
                                   self.C_ACTIONS)

    def play(self, game_state, observation):

        self._new_state = {'x': None,
                           'y': None,
                           'game_state': game_state,
                           'observation': observation}

        # GET NEW INPUTS
        x = self.get_env_state()

        # THINK THE NEXT MOVE
        y = self.think(x)

        # GET NEW NEW ACTION
        actions = self.get_agent_action(y)

        # UPDATE PREVIOUS STATE
        self.update_memory(x, y, actions)

        return actions

    def init_memory(self, game_state, observation):
        """
        Init memory
        """

        self._new_state = {'x': None,
                           'y': None,
                           'actions': [],
                           'game_state': game_state,
                           'observation': observation}

        output_shape = len(self.C_ACTIONS + self.W_ACTIONS)
        x = self.get_env_state()
        y = np.zeros((x.shape[0], x.shape[1], output_shape))

        if self._epsilon > self._epsilon_final:
            self._epsilon = self._epsilon * self._epsilon_decay

        self._new_state['x'] = x
        self._new_state['y'] = y
        self._reward.init()
        self._old_state = self._new_state

    def update_memory(self, x, y, actions):
        """
        Update memory
        """

        self._new_state['x'] = x
        self._new_state['y'] = y
        self._new_state['actions'] = actions

        # LEARN FROM THE LAST REWARD
        self.learn()

        self._old_state = self._new_state

    def think(self, x):
        """
        Model predict
        """

        y = self._model.predict(x)

        if random.random() < self._epsilon:
            y = np.random.rand(*y.shape)

        return y

    def learn(self):
        """
        Model train
        """

        old_x = self._old_state['x']
        old_y = self._old_state['y']

        # FIT MODEL
        new_y = self._reward.update(self._new_state,
                                    self._old_state)

        if len(self._reward._memory) >= self._batch_length:
            x_batch, y_batch = self._reward.get_batch()
            if x_batch != [] and y_batch != []:
                x_batch, y_batch = self._data_augmentor.get_batch(
                    x_batch, y_batch)
                self._model.fit(x_batch, y_batch, epochs=self._epochs)
            self._reward.init()

    def get_env_state(self):
        """
        TRansform game state into model input matrix
        """

        game_state = self._new_state['game_state']
        observation = self._new_state['observation']

        player = game_state.players[observation.player]
        opponent = game_state.players[(observation.player + 1) % 2]

        # MAP SHAPE
        w, h = game_state.map.width, game_state.map.height

        resource_map = {"wood": 1, "coal": 2, "uranium": 3}
        r = [
            [0 if game_state.map.map[j][i].resource is None
             else resource_map[game_state.map.map[j][i].resource.type]
             for i in range(h)] for j in range(w)
        ]

        # MAP UNITS
        capacity = GAME_CONSTANTS["PARAMETERS"]["RESOURCE_CAPACITY"]["WORKER"]
        shape = (w, h, 6)
        u = np.zeros(6 * w * h).reshape(*shape) - 1
        units = player.units + opponent.units
        for i in units:
            u[i.pos.y][i.pos.x] = [i.type,
                                   i.team,
                                   i.cooldown,
                                   i.cargo.wood / capacity,
                                   i.cargo.coal / capacity,
                                   i.cargo.uranium / capacity]

        # CITIES IN MAP
        e = list(player.cities.values())
        e += list(opponent.cities.values())

        shape = (w, h, 4)
        c = np.zeros(4 * w * h).reshape(*shape) - 1
        for city in e:
            citytiles = city.citytiles
            for i in citytiles:
                c[i.pos.y][i.pos.x] = [i.cooldown,
                                       city.fuel / 100,
                                       city.light_upkeep,
                                       city.team]

        return np.dstack([r, u, c])

    def get_agent_action(self, new_y):
        """
        Transform model output into game actions
        """

        actions = []

        game_state = self._new_state['game_state']
        observation = self._new_state['observation']
        player = game_state.players[observation.player]

        # SPLIT UNIT AND CITY PREDICTIONS
        y_unit = new_y[:, :, 0:len(self.W_ACTIONS)]
        y_city = new_y[:, :, len(self.W_ACTIONS):]

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
