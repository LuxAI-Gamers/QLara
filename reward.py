from abc import ABC
from abc import abstractmethod

import numpy as np


class Reward():

    def __init__(self):
        self.init()

    def init(self):
        self._memory = []

    def update(self, new_state, old_state):
        self._memory.append((new_state, old_state))

    @abstractmethod
    def reward_function(self, new_reward, old_reward):
        pass

    @abstractmethod
    def get_target(self):
        pass


class BatchReward(Reward):

    def __init__(self, gamma, lr, W_ACTIONS, C_ACTIONS):

        super(BatchReward, self).__init__()

        self._lr = lr
        self._gamma = gamma
        self.W_ACTIONS = W_ACTIONS
        self.C_ACTIONS = C_ACTIONS

    def reward_function(self, new_state, old_state):
        """
        """
        #old_reward = old_state['observation']['reward']
        #new_reward = new_state['observation']['reward']

        #reward = 1 if new_reward >= old_reward else -0.2
        #reward = 0 if new_reward == old_reward else reward

        game_state = new_state['game_state']
        observation = new_state['observation']

        player = game_state.players[observation.player]
        opponent = game_state.players[(observation.player + 1) % 2]
        
        r_player = sum([len(city.citytiles) for city in player.cities.values()])
        r_opponent = sum([len(city.citytiles) for city in opponent.cities.values()])

        r_player = r_player*10 + len(player.units)
        r_opponent = r_player*10 + len(opponent.units)
                
        reward = 1 if r_player >= r_opponent else -0.2
        reward = 0 if r_player == r_opponent else reward
        
        return reward

    
    def validate_actions(self, new_state, old_state, reward):

        actions = old_state['actions']

        old_game_state = old_state['game_state']
        old_observation = old_state['observation']

        new_game_state = new_state['game_state']
        new_observation = new_state['observation']

        old_player = old_game_state.players[old_observation.player]
        new_player = new_game_state.players[new_observation.player]

        units_that_acted = [a for a in actions if 'u_' in a]
        units_that_acted = [a.split(' ')[1] for a in units_that_acted]

        for new_unit in new_player.units:
            for old_unit in old_player.units:
                if new_unit.id == old_unit.id and \
                        new_unit.pos.x == old_unit.pos.x and \
                        new_unit.pos.y == old_unit.pos.y and \
                        new_unit.cargo.wood == old_unit.cargo.wood and \
                        new_unit.cargo.coal == old_unit.cargo.coal and \
                        new_unit.cargo.uranium == old_unit.cargo.uranium:

                    if new_unit.id in units_that_acted:
                        reward[new_unit.pos.y, new_unit.pos.x] = -1

        cities_that_acted = [a for a in actions if a[0:2] in ['bw','r ']]
        cities_that_acted = [[int(a.split(' ')[2]), int(a.split(' ')[1])]
                              for a in cities_that_acted]

        for city in new_player.cities.values():
            for citytile in city.citytiles:
                if citytile.cooldown==0:
                    if [citytile.pos.y, citytile.pos.x] in cities_that_acted:
                        reward[citytile.pos.y, citytile.pos.x] = -1
        return reward


    def correct_old_prediction(self, new_state, old_state, reward):

        old_y = old_state['y']
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

    def get_batch(self):

        first_state = self._memory[0][0]
        last_state = self._memory[-1][-1]
        batch_reward = self.reward_function(first_state, last_state)

        x_batch = []
        y_batch = []
        for new_state, old_state in self._memory:

            old_x = old_state['x']
            old_y = old_state['y']
            new_y = new_state['y']

            reward_matrix = batch_reward + self._gamma * np.amax(old_y, axis=2)
            
            reward_matrix = self.validate_actions(
                new_state, old_state, reward_matrix)
            reward_matrix = self.correct_old_prediction(
                new_state, old_state, reward_matrix)

            reward_matrix = (1 - self._lr) * old_y + self._lr * reward_matrix

            x_batch.append(old_x)
            y_batch.append(reward_matrix)

        self._memory = []

        return x_batch, y_batch
