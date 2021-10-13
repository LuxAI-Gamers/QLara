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
    'episodes': 25
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

    episodes = configuration['episodes']
    configuration.pop('episodes', None)

    clara = Clara(**configuration)

    for ep in range(episodes):
        clear_output()
        print(f"=== Episode {ep} ===")
        env = make("lux_ai_2021",
                   configuration={"seed": random.randint(0, 99999999),
                                  "loglevel": 1,
                                  "annotations": True},
                   debug=True)

        steps = env.run([agent, "simple_agent"])
    print([step[0]['action'] for step in steps])

    clara._model.save('models')

    replay = env.toJSON()
    with open("replay.json", "w") as f:
        json.dump(replay, f)
