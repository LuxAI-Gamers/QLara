import random

from lux.game import Game

from qlara import QLara

configuration = {
    'lr': 0.01,
    'gamma': 0.9,
    'epsilon': 0.9,
    'epsilon_final': 0.5,
    'epsilon_decay': 0.995,
    'batch_length': 1,
    'epochs': 0,
    'episodes': 0
}

episodes = configuration.pop('episodes')

qlara = QLara(**configuration)


def agent(observation):
    global game_state, clara

    ### Do not edit ###
    if observation["step"] == 0:
        game_state = Game()
        game_state._initialize(observation["updates"])
        game_state._update(observation["updates"][2:])
        game_state.id = observation.player
        qlara._model.load('qlara/models')
        qlara.init_memory(game_state, observation)
        actions = []
    else:
        game_state._update(observation["updates"])

        ### AI Code goes down here! ###
        actions = qlara.play(game_state, observation)

    return actions


if __name__ == '__main__':

    import json
    import random
    from kaggle_environments import make
    from IPython.display import clear_output

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
