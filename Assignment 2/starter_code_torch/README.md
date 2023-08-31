# RL with MinAtar

## Install
Install either Anaconda or Miniconda using instructions below

https://docs.conda.io/projects/continuumio-conda/en/latest/user-guide/install/index.html

After installing Conda, follow the following instructions on a terminal:
```bash
cd <assignment_dir>
Create a conda environment using the following:

conda env create -f cs234-torch-<your-system>.yml
conda activate cs234-torch

pip install -r requirements.txt
git clone https://github.com/kenjyoung/MinAtar.git
cd MinAtar
pip install .
cd ../
```

NOTE: If you are using an M1-M2 Mac, you might run into trouble installing a package called
grpcio. If so, we recommend installing it from this repo:
https://github.com/pietrodn/grpcio-mac-arm-build/releases.
Download grpcio-1.51.1-cp39-cp39-macosx_11_0_arm64.whl from the releases, and run pip install grpcio-1.51.1-cp39-cp39-macosx_11_0_arm64.whl

## Environment

### MinAtar/Breakout

- The player controls a bar that can move horizontally, and gets rewards by bouncing a ball into bricks.

```python
# action = int in [0, 6)
# state  = (10, 10, 4) boolean array
# reward = 1 when agent breaks a brick, 0 every other step
```

## Training

Once done with implementing `q4_linear_torch.py` and `q5_nature_torch` make sure you test your implementation by launching `python q4_linear_torch.py` and `python q5_nature_torch.py` that will run your code on the Test environment.

You can launch the training of DQN on breakout with

```
python q6_train_atari_nature.py
```


Training tips: 
(1) The starter code writes summaries of a bunch of useful variables that can help you monitor the training process.
You can monitor your training with Tensorboard by doing, on Azure

```
tensorboard --logdir=results
```

and then connect to `ip-of-you-machine:6006`


(2) You can use ‘screen’ to manage windows on VM and to retrieve running programs. 
Before training DQN on Atari games, run 

```
screen 
```
then run 

```
python q6_train_atari_nature.py
```
By using Screen, programs continue to run when their window is currently not visible and even when the whole screen session is detached 
from the users terminal. 

To detach from your window, simply press the following sequence of buttons

```
ctrl-a d
```
This is done by pressing control-a first, releasing it, and press d


To retrieve your running program on VM, simply type

```
screen -r
```
which will recover the detached window.   



**Credits**
Assignment code written by Guillaume Genthial and Shuhui Qu.
Assignment code updated by Jian Vora and Max Sobol Mark
