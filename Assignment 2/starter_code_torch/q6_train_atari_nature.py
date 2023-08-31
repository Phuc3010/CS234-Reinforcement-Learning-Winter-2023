import gym
from minatar import Environment
import subprocess
from pathlib import Path

from q3_schedule import LinearExploration, LinearSchedule
from q5_nature_torch import NatureQN

from configs.q6_train_atari_nature import config
from utils.general import export_mean_plot
import logging

"""
Use deep Q network for the Atari game. Please report the final result.
Feel free to change the configurations (in the configs/ folder). 
If so, please report your hyperparameters.

You'll find the results in the results folder. A good way to monitor the progress
of the training is to use Tensorboard. The starter code writes summaries of different
variables.

To launch tensorboard, open a Terminal window and run 
tensorboard --logdir=results/
Then, connect remotely to 
address-ip-of-the-server:6006 
6006 is the default port used by tensorboard.
"""
if __name__ == "__main__":
    logging.getLogger(
        "matplotlib.font_manager"
    ).disabled = True  # disable font manager warnings
    # make env
    env = Environment("breakout")
    num_runs = 3

    for i in range(num_runs):
        # exploration strategy
        exp_schedule = LinearExploration(
            env, config.eps_begin, config.eps_end, config.eps_nsteps
        )

        # learning rate schedule
        lr_schedule = LinearSchedule(config.lr_begin, config.lr_end, config.lr_nsteps)

        # train model
        model = NatureQN(env, config)
        model.run(exp_schedule, lr_schedule, run_idx=i + 1)

    export_mean_plot("Scores", config.plot_output, config.output_path)
