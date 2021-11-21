# Hi! I'm QLara.

Hello everyone! My name is QLara. I am a LuxAI agent gamer based in Reinforcement Learning. I'm here to show you how I am programmed and to get you started on Q-Learning.

<p align="center">
  <img width="400" src="img/avatar/clara_400px.jpg">
</p>


Before going any further into myself, I'm programmed to play the [Lux Kaggle Competition](https://www.lux-ai.org/). You can read all competition rules, this may be a bit of a learning curve and you can skip it if you are only interested _in me_ but make sure to read it before modify my behaviour. 


## What will you find in this project?

A Reinforcement Learning agent to be trained to play LuxAI Challenge. Our trained model is not included in the repository. But you can train it on your own, in your local machine using the script [agent.py](agent.py)

```sh
$ python agent.py
```
You can configure the training parameters in this file by editing the `configuration`variable:

```python
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
```


## Play the game!

Do you want to play a game?  The [Lux repository](https://github.com/Lux-AI-Challenge/Lux-Design-2021#getting-started) has a detailed guide on installing dependencies. Here are the main steps:

- python >=3.7
- nodeJS >=12
- luxAI node package. Installation: 

```sh
$ npm install -g @lux-ai/2021-challenge@latest
```
- all packages in `requierements.txt`. Installation: 

```sh
$ pip install -r requirements.txt
```

After installing all dependencies open the `visualizer.ipynb` to run and watch games!

```sh
$ jupyter notebook visualizer.ipynb
```



This will give you the visualizer to watch the game and minimally interact with the play such as this

<p align="center">
  <img width="600" height="473" src="img/game.gif">
</p>

The `seed` parameter will give you different maps and you can see how the map affects the implemented behaviour.


```python
env = make("lux_ai_2021", configuration={"seed": 666, "loglevel": 0, "annotations": True}, debug=True)
env.render(mode="ipython", width=1000, height=800)
```

## Play using Docker!

Of course, you can ignore all issues with dependencies if you have Docker installed. You can jump directly into visualizing a game.

First, you'll need to build the image.
```bash
docker build -t qlara .
```

Now, you can launch Jupyter in your browser by running a container.
```bash
docker run -v $(pwd):/root -p 8888:8888 -it --rm qlapa:latest
```

## References (Spanish talk)

* https://youtu.be/NiixJV3c7f4?t=3878
