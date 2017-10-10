# universe-starter-agent

The codebase implements a starter agent that can solve a number of `universe` environments.
It contains a basic implementation of the [A3C algorithm](https://arxiv.org/abs/1602.01783), adapted for real-time environments.

# Dependencies

* Python 2.7 or 3.5
* [Golang](https://golang.org/doc/install)
* [six](https://pypi.python.org/pypi/six) (for py2/3 compatibility)
* [TensorFlow](https://www.tensorflow.org/) 0.12
* [tmux](https://tmux.github.io/) (the start script opens up a tmux session with multiple windows)
* [htop](https://hisham.hm/htop/) (shown in one of the tmux windows)
* [gym](https://pypi.python.org/pypi/gym)
* gym[atari]
* libjpeg-turbo (`brew install libjpeg-turbo`)
* [universe](https://pypi.python.org/pypi/universe)
* [opencv-python](https://pypi.python.org/pypi/opencv-python)
* [numpy](https://pypi.python.org/pypi/numpy)
* [scipy](https://pypi.python.org/pypi/scipy)

# Getting Started

```
conda create --name universe-starter-agent python=3.5
source activate universe-starter-agent

brew install tmux htop cmake golang libjpeg-turbo      # On Linux use sudo apt-get install -y tmux htop cmake golang libjpeg-dev

pip install "gym[atari]"
pip install universe
pip install six
pip install tensorflow
conda install -y -c https://conda.binstar.org/menpo opencv3
conda install -y numpy
conda install -y scipy
```


Add the following to your `.bashrc` so that you'll have the correct environment when the `train.py` script spawns new bash shells
```source activate universe-starter-agent```

## Atari Pong

`python train.py --num-workers 2 --env-id PongDeterministic-v3 --log-dir /tmp/pong`

The command above will train an agent on Atari Pong using ALE simulator.
It will see two workers that will be learning in parallel (`--num-workers` flag) and will output intermediate results into given directory.

The code will launch the following processes:
* worker-0 - a process that runs policy gradient
* worker-1 - a process identical to process-1, that uses different random noise from the environment
* ps - the parameter server, which synchronizes the parameters among the different workers
* tb - a tensorboard process for convenient display of the statistics of learning

Once you start the training process, it will create a tmux session with a window for each of these processes. You can connect to them by typing `tmux a` in the console.
Once in the tmux session, you can see all your windows with `ctrl-b w`.
To switch to window number 0, type: `ctrl-b 0`. Look up tmux documentation for more commands.

To access TensorBoard to see various monitoring metrics of the agent, open [http://localhost:12345/](http://localhost:12345/) in a browser.

Using 16 workers, the agent should be able to solve `PongDeterministic-v3` (not VNC) within 30 minutes (often less) on an `m4.10xlarge` instance.
Using 32 workers, the agent is able to solve the same environment in 10 minutes on an `m4.16xlarge` instance.
If you run this experiment on a high-end MacBook Pro, the above job will take just under 2 hours to solve Pong.

Add '--visualise' toggle if you want to visualise the worker using env.render() as follows:

`python train.py --num-workers 2 --env-id PongDeterministic-v3 --log-dir /tmp/pong --visualise`

![pong](https://github.com/openai/universe-starter-agent/raw/master/imgs/tb_pong.png "Pong")

For best performance, it is recommended for the number of workers to not exceed available number of CPU cores.

You can stop the experiment with `tmux kill-session` command.

## Playing games over remote desktop

The main difference with the previous experiment is that now we are going to play the game through VNC protocol.
The VNC environments are hosted on the EC2 cloud and have an interface that's different from a conventional Atari Gym
environment;  luckily, with the help of several wrappers (which are used within `envs.py` file)
the experience should be similar to the agent as if it was played locally. The problem itself is more difficult
because the observations and actions are delayed due to the latency induced by the network.

More interestingly, you can also peek at what the agent is doing with a VNCViewer.

Note that the default behavior of `train.py` is to start the remotes on a local machine. Take a look at https://github.com/openai/universe/blob/master/doc/remotes.rst for documentation on managing your remotes. Pass additional `-r` flag to point to pre-existing instances.

### VNC Pong

`python train.py --num-workers 2 --env-id gym-core.PongDeterministic-v3 --log-dir /tmp/vncpong`

_Peeking into the agent's environment with TurboVNC_

You can use your system viewer as `open vnc://localhost:5900` (or `open vnc://${docker_ip}:5900`) or connect TurboVNC to that ip/port.
VNC password is `"openai"`.

![pong](https://github.com/openai/universe-starter-agent/raw/master/imgs/vnc_pong.png "Pong over VNC")

#### Important caveats

One of the novel challenges in using Universe environments is that
they operate in *real time*, and in addition, it takes time for the
environment to transmit the observation to the agent.  This time
creates a lag: where the greater the lag, the harder it is to solve
environment with today's RL algorithms.  Thus, to get the best
possible results it is necessary to reduce the lag, which can be
achieved by having both the environments and the agent live
on the same high-speed computer network.  So for example, if you have
a fast local network, you could host the environments on one set of
machines, and the agent on another machine that can speak to the
environments with low latency.  Alternatively, you can run the
environments and the agent on the same EC2/Azure region.  Other
configurations tend to have greater lag.

To keep track of your lag, look for the phrase `reaction_time` in
stderr.  If you run both the agent and the environment on nearby
machines on the cloud, your `reaction_time` should be as low as 40ms.
The `reaction_time` statistic is printed to stderr because we wrap our
environment with the `Logger` wrapper, as done in
[here](<https://github.com/openai/universe-starter-agent/blob/master/envs.py#L32>).

Generally speaking, environments that are most affected by lag are
games that place a lot of emphasis on reaction time.  For example,
this agent is able to solve VNC Pong
(`gym-core.PongDeterministic-v3`) in under 2 hours when both the agent
and the environment are co-located on the cloud, but this agent had
difficulty solving VNC Pong when the environment was on the cloud
while the agent was not.  This issue affects environments that place
great emphasis on reaction time.

### A note on tuning

This implementation has been tuned to do well on VNC Pong, and we do not guarantee
its performance on other tasks.  It is meant as a starting point.

### Playing flash games

You may run the following command to launch the agent on the game Neon Race:

`python train.py --num-workers 2 --env-id flashgames.NeonRace-v0 --log-dir /tmp/neonrace`

_What agent sees when playing Neon Race_
(you can connect to this view via [note](#vnc-pong) above)
![neon](https://github.com/openai/universe-starter-agent/raw/master/imgs/neon_race.png "Neon Race")

Getting 80% of the maximal score takes between 1 and 2 hours with 16 workers, and getting to 100% of the score
takes about 12 hours.  Also, flash games are run at 5fps by default, so it should be possible to productively
use 16 workers on a machine with 8 (and possibly even 4) cores.

### Next steps

Now that you have seen an example agent, develop agents of your own.  We hope that you will find
doing so to be an exciting and an enjoyable task.

# ADDITIONAL README
# High performance GRU-A3C agent

- 80% performance increase over the current LSTM-based A3C starter agent on `PongDeterministic-v3` environment.

- 20% higher throughput in all Atari environments.

- 10-100% data-efficiency increase in various Atari environments. This boost is not well-characterized or well-tuned yet.

- 3 small, initial hyperparameter changes extend these gains even more.

- GRU-based RNNs appear both more stable and more tolerant of a wider hyperparameter space than LSTMs for discrete RL tasks.

- Research cycles were shortened 30% more by MKL-optimized [`manjaro-ml-dotfiles`](https://github.com/louiehelm/manjaro-ml-dotfiles). This improves upon the recommended `universe` dev environment of `ubuntu` + `anaconda`.


## New GRU-A3C Agent (~3x faster @ Atari Pong)


`python train.py --num-workers 8 --env-id PongDeterministic-v3 --visualise --log-dir /tmp/pong`

GRU-A3C can solve `PongDeterministic-v3` in under **30 minutes** on a high-end Mac (vs **2 hours** for the original starter agent). To solve `PongDeterministic-v3` in 30 minutes before, the LSTM-based agent required 2x as many agents, runing on 4x as many cores, all at higher clock speeds.

![pong](http://i.imgur.com/QphU60g.png "GRU Pong")


This smaller pool of agents can also solve `PongDeterministic-v3` with fewer total frames (~900k vs 2.5-3.2M). Data efficiency gains from GRU-based A3C agents need more characterization but definitely appear significant.


# Major Changes

* Replace LSTMs in A3C model with [more principled GRUs](http://colah.github.io/posts/2015-08-Understanding-LSTMs/)
* RNN switched to time-major data format to avoid two extra transposes per step


# Minor Changes

* 100% higher learning rate with slower decay
* Local steps increased 20 -> 30
* 10% dropout
* Asyncronous visualiser


GRUs appear less overwhelmed by higher learning rates and more resilient to a wider range of hyperparams in general.

Local steps can scale both lower and higher with GRU-based agents:

- 15 provides less actions per second but higher data efficiency

- 30 provides more actions per second with slightly worse data efficiency



These hyperparameter changes are not exhaustively tuned, nor the prime drivers of higher performance. They mostly serve to demonstrate a new set of learning parameters that was previously unworkable with LSTMs but which appears both well-behaved and high performance with GRUs in many environments.



## Terminal Visualiser

![vis](http://i.imgur.com/oFuBqJP.jpg "Terminal Visualiser")

Display 2 running agents: `./term_vis.sh 2` 

There is also a small code patch to allow asyncronous visualisation via bash script (requires w3m-img)

* Visualisation can be switched on / off even on live training runs
* Uses 60% less CPU per frame vs pyglet when actively displaying
* Workers spend 0% CPU updating visualisation unless  the script is actually running
* Can visualise over SSH via X-Forwarded terminals (e.g., ssh -L localhost:12345:localhost:12345 -Ycv -i <.pem file> ubuntu@xxx.xxx.xxx.xxx) [see above]
