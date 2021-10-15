from datetime import datetime
import logging
import random

from lux.game import Game

from clarai import Clara

configuration = {
    'lr': 0.01,
    'gamma': 0.95,
    'epsilon': 0.95,
    'epsilon_final': 0.01,
    'epsilon_decay': 0.995,
    'batch_length': 12,
    'epochs': 1,
    'episodes': 1
}

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


if __name__ == '__main__':

    import json
    import random
    from kaggle_environments import make
    from IPython.display import clear_output

    # Set logger
    logger = logging.Logger('ClarAI')
    fh = logging.FileHandler(f'{datetime.now().strftime("%Y-%m-%d-%X")}.log')
    fh.setLevel(logging.INFO)
    fh.setFormatter(logging.Formatter("%(asctime)s %(name)s: %(levelname)s:: %(message)s"))
    logger.addHandler(fh)

    logger.info('===== Welcome to ClarAI =====')
    logger.info(configuration)

    episodes = configuration['episodes']
    configuration.pop('episodes')

    clara = Clara(**configuration, logger=logger)

    for ep in range(episodes):
        logger.info(f"=== Episode {ep} ===")
        clear_output()
        env = make("lux_ai_2021",
                   configuration={"seed": random.randint(0, 99999999),
                                  "loglevel": 1,
                                  "annotations": True},
                   debug=True)
        
        env.run([agent, "simple_agent"])
        if ep % 1000 == 0 and ep > 0:
            logger.info(json.dumps(env.toJSON()))
        else:
            # Create mini dict with info
            mini_env = {
                'id': env.id,
                'rewards': [env.state[0]['reward'], env.state[1]['reward']],
                'seed': env.configuration.seed,
                'height': env.steps[0][0]['observation']['height'],
                'width': env.steps[0][0]['observation']['width'],
                'actions': [step[0]['action'] for step in env.steps],
                'cities': [
                    [{
                        'CityCount': step[0]['observation']['globalCityIDCount'],
                        'UnitCount': step[0]['observation']['globalUnitIDCount']
                    }]
                    for step in env.steps
                ]
            }
            logger.info(json.dumps(mini_env))
           
    clara._model.save('models')
        
