import os
import json
import random
from kaggle_environments import make
from IPython.display import clear_output

from agent import agent, clara

args = {
    'episodes': 25,
    'model_dir': './models'
}

if __name__ == '__main__':
    ### HYPERPARAMETERS ###

    if os.path.isfile('/opt/ml/input/config/hyperparameters.json'):
        # This is where SageMaker puts the hyperparameters
        # I would love to use argparse, but that's not how SageMakers developers have made it:
        # https://github.com/aws/sagemaker-training-toolkit/issues/65
        with open('/opt/ml/input/config/hyperparameters.json', 'r') as f:
            sagemaker_args = json.load(f)

        args.update(sagemaker_args)

    episodes = int(args["episodes"])
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

    clara._model.save(args["model_dir"])
    print(os.listdir(args["model_dir"]))

    replay = env.toJSON()
    with open("replay.json", "w") as f:
        json.dump(replay, f)