import math
import sys
import random

from lux.game import Game
from lux.game_map import Cell, RESOURCE_TYPES, Position
from lux.game_objects import Unit
from lux.constants import Constants
from lux.game_constants import GAME_CONSTANTS
from lux import annotate

from clarai import Clara




clara = Clara()


def agent(observation, configuration):
    global game_state, clara

    ### Do not edit ###
    if observation["step"] == 0:
        game_state = Game()
        game_state._initialize(observation["updates"])
        game_state._update(observation["updates"][2:])
        game_state.id = observation.player
        clara.init_old_state(game_state, observation)
        actions = []
    else:
        game_state._update(observation["updates"])


    ### AI Code goes down here! ###
        actions = clara.play(game_state, observation)

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
                                "loglevel": 1,
                                "annotations": True},
                debug=True)

        steps = env.run([agent, "simple_agent"])
    print([step[0]['action'] for step in steps])

    replay = env.toJSON()
    with open("replay.json", "w") as f:
        json.dump(replay, f)
