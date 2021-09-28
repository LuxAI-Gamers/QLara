from abc import ABC
from abc import abstractmethod

import numpy as np


class Reward():

    @abstractmethod
    def reward_function(self, new_reward, old_reward):
        pass

    def get_target(self, new_state, old_state):
        pass


class CustomReward(Reward):

    def __init__(self, gamma, W_ACTIONS, C_ACTIONS):

        self._gamma = gamma
        self.W_ACTIONS = W_ACTIONS
        self.C_ACTIONS = C_ACTIONS

    def reward_function(self, new_reward, old_reward):
        """
        """
        reward = new_reward / old_reward - 1 if old_reward != 0 else 0.1

        return reward

    def compute_reward(self, new_state, old_state):

        old_y = old_state['y']
        old_reward = old_state['observation']['reward']
        new_reward = new_state['observation']['reward']

        reward = self.reward_function(new_reward, old_reward)
        reward_matrix = reward + self._gamma * np.amax(old_y, axis=2)

        return reward_matrix

    def validate_actions(self, new_state, old_state, reward):

        actions = old_state['actions']

        old_game_state = old_state['game_state']
        old_observation = old_state['observation']

        new_game_state = new_state['game_state']
        new_observation = new_state['observation']

        old_player = old_game_state.players[old_observation.player]
        new_player = new_game_state.players[new_observation.player]

        units_that_acted = [action.split(' ')[1] for action in actions]
        for new_unit in new_player.units:
            for old_unit in old_player.units:
                if new_unit.id == old_unit.id and \
                        new_unit.pos.x == old_unit.pos.x and \
                        new_unit.pos.y == old_unit.pos.y and \
                        new_unit.cargo.wood == old_unit.cargo.wood and \
                        new_unit.cargo.coal == old_unit.cargo.coal and \
                        new_unit.cargo.uranium == old_unit.cargo.uranium:

                    if new_unit.id in units_that_acted:
                        reward[new_unit.pos.y, new_unit.pos.x] = -2

        return reward

    def correct_old_prediction(self, new_state, old_state, reward):

        old_x = old_state['x']
        old_y = old_state['y']

        new_x = new_state['x']
        new_y = new_state['y']

        game_state = old_state['game_state']
        observation = old_state['observation']

        y_unit = new_y[:, :, 0:len(self.W_ACTIONS)]
        y_city = new_y[:, :, len(self.W_ACTIONS):-1]

        player = game_state.players[observation.player]

        # CITY LOSS
        best_actions_map = np.argmax(y_city, axis=2)
        for city in player.cities.values():
            for city_tile in city.citytiles:
                j, i = city_tile.pos.x, city_tile.pos.y
                idx = best_actions_map[i, j]
                old_y[i, j][idx] = reward[i, j]

        # UNITS LOSS
        best_actions_map = np.argmax(y_unit, axis=2)
        for unit in player.units:
            j, i = unit.pos.x, unit.pos.y
            idx = best_actions_map[i, j]
            old_y[i, j][idx] = reward[i, j]

        return old_y

    def get_target(self, new_state, old_state):

        reward = self.compute_reward(new_state, old_state)
        reward = self.validate_actions(new_state, old_state, reward)
        target = self.correct_old_prediction(new_state, old_state, reward)

        return target
