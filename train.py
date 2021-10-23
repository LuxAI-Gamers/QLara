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


def convert_to_proper_type(str_v):
    '''
    Receive a string variable and return it as float, integer or string.
    '''
    try:
        float_v = float(str_v)
        int_v = int(float_v)
    except:
        return str_v  
    if (float_v == int_v): return int_v
    return float_v


configuration = {
    'lr': 0.5,
    'gamma': 0.95,
    'epsilon': 0.5,
    'epsilon_final': 0.01,
    'epsilon_decay': 0.995,
    'batch_length': 12,
    'epochs': 1,
    'episodes': 1000,
    'model_dir': './models',
    'games_dir': './games'
}


if __name__ == '__main__':
    # Read hyperparameters from Sagemaker if applicable
    if os.path.isfile('/opt/ml/input/config/hyperparameters.json'):
        with open('/opt/ml/input/config/hyperparameters.json', 'r') as f:
            sagemaker_args = json.load(f)

        # Update values with sagemakers hyperparameters
        configuration.update(sagemaker_args)
        # Convert hyperparameters from string to their proper type
        configuration = {k: convert_to_proper_type(v) for k, v in configuration.items()}

    print(f'Training configuration: {configuration}')

    # Extract external parameters from Clarai configuration
    episodes = int(configuration.pop('episodes'))
    model_dir = configuration.pop('model_dir')
    games_dir = configuration.pop('games_dir')
    os.makedirs(games_dir, exist_ok=True)

    # Create Clarai agent with configuration
    clara = Clara(**configuration)

    # Play and learn
    for ep in range(episodes):
        clear_output()
        print(f"==== Episode {ep} ====")
        env = make("lux_ai_2021",
                   configuration={"seed": 7,
                                  "loglevel": 1,
                                  "annotations": True},
                   debug=True)

        env.run([agent, "simple_agent"])

        # Print metrics
        rewards = [env.state[0]['reward'], env.state[1]['reward']]
        winner = max(rewards)
        table_data = [
            ['id::', env.id],
            ['seed::', env.configuration.seed],
            ['winner::', rewards.index(winner)],
            ['board::', env.steps[0][0]['observation']['width']],
            ['rounds::', len(env.steps)],
            ['units::', env.steps[-1][0]['observation']['globalUnitIDCount']],
            ['cities::', env.steps[-1][0]['observation']['globalCityIDCount']]
        ]
        for row in table_data:
            print("{}{};".format(*row))

        # Save model in this episodes:
        episodes_to_save = [
            1,
            2,
            100,
            200,
            500,
            1000,
            2000,
            3000,
            4000,
            5000,
            7500,
            10000,
            episodes - 1]
        if ep in episodes_to_save:
            clara._model.save(model_dir)
            with open(f"{games_dir}/replay_{ep}.json", "w") as f:
                json.dump(env.toJSON(), f)
