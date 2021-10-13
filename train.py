import json
import random
import argparse
from kaggle_environments import make
from IPython.display import clear_output

from agent import agent, clara

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--episodes', type=int, default=25)
    parser.add_argument('--model_dir', type=str, default='./models')
    args = parser.parse_args()

    for ep in range(args.episodes):
        clear_output()
        print(f"=== Episode {ep} ===")
        env = make("lux_ai_2021",
                   configuration={"seed": random.randint(0, 99999999),
                                  "loglevel": 1,
                                  "annotations": True},
                   debug=True)

        steps = env.run([agent, "simple_agent"])
    print([step[0]['action'] for step in steps])

    clara._model.save(args.model_dir)

    replay = env.toJSON()
    with open("replay.json", "w") as f:
        json.dump(replay, f)