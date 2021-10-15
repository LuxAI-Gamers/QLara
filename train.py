import os
import json
import random
from kaggle_environments import make
from IPython.display import clear_output

from lux.game import Game
from clarai import Clara


def agent(observation):
    global game_state, clara

    ### Do not edit ###
    if observation["step"] == 0:
        game_state = Game()
        game_state._initialize(observation["updates"])
        game_state._update(observation["updates"][2:])
        game_state.id = observation.player
        # clara._model.load('models/1632520074/')
        clara.init_memory(game_state, observation)
        actions = []
    else:
        game_state._update(observation["updates"])

        ### AI Code goes down here! ###
        actions = clara.play(game_state, observation)

    return actions


configuration = {
    'lr': 0.01,
    'gamma': 0.95,
    'epsilon': 0.95,
    'epsilon_final': 0.01,
    'epsilon_decay': 0.995,
    'batch_length': 12,
    'epochs': 1,
    'episodes': 25,
    'model_dir': './models'
}


if __name__ == '__main__':
    # Read hyperparameters from Sagemaker if applicable
    if os.path.isfile('/opt/ml/input/config/hyperparameters.json'):
        with open('/opt/ml/input/config/hyperparameters.json', 'r') as f:
            sagemaker_args = json.load(f)

        configuration.update(sagemaker_args)

    print(configuration)

    # Extract external parameters from Clarai configuration
    episodes = int(configuration.pop('episodes'))
    model_dir = configuration.pop('model_dir')

    # Create Clarai agent with configuration
    clara = Clara(**configuration)

    # Play and learn
    for ep in range(episodes):
        clear_output()
        print(f"=== Episode {ep} ===")
        env = make("lux_ai_2021",
                   configuration={"seed": random.randint(0, 99999999),
                                  "loglevel": 1,
                                  "annotations": True},
                   debug=True)

        steps = env.run([agent, "simple_agent"])

        # Print metrics
        total_rounds = len(steps)
        units_created = steps[-1][0]['observation']['globalUnitIDCount']
        cities_created = steps[-1][0]['observation']['globalCityIDCount']
        print(
            f'total_rounds={total_rounds}; units_created={units_created}; cities_created={cities_created};')

    # Save model
    clara._model.save(model_dir)

    replay = env.toJSON()
    with open("replay.json", "w") as f:
        json.dump(replay, f)
