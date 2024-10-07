# Planning with Learning

In autonomous driving, end-to-end methods utilizing Imitation Learning (IL) and Reinforcement Learning (RL) are becoming more and more common. 
However, they do not involve explicit reasoning like classic robotics workflow and planning with horizons, resulting in strategies implicit and myopic. 

In this paper, we introduce a path planning method that uses 
Behavioral Cloning (BC) for path-tracking and Proximal Policy Optimization (PPO) for static obstacle nudging. 
It outputs lateral offset values to adjust the given reference waypoints and performs modified path for different controllers. 

Experimental results show that the algorithm can do path following that mimics the expert performance of path-tracking controllers, and avoid collision to fixed obstacles. The method makes a good attempt at planning with learning-based methods in path planning problems of autonomous driving. 

## Setup

Install [Anaconda](https://docs.anaconda.com/free/anaconda/install/index.html), [f1tenth_gym](https://github.com/f1tenth/f1tenth_gym), and project dependencies.
Check [gym 0.21 issue](https://github.com/openai/gym/issues/3176) for further details.
**Configuration is subject to change.**
```bash
conda create -n ese-650-fp python=3.8 pip=21 setuptools=65.5.0  # create a new conda env under the root dir
conda activate ese-650-fp
pip install -e .  # install setup.py for f1tenth_gym
pip install psutil packaging  # install ignored ipykernel dependencies
pip install -r requirements.txt  # install other dependencies
# conda deactivate  # exit ese-650-fp env
# conda remove -n ese-650-fp --all  # remove ese-650-fp env
```

Install pytorch and Cuda. Please check [official installation](https://pytorch.org/get-started/locally/) for **your** platforms.
```bash
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118  # for Derek's PC
```

To play around [cleanrl](https://github.com/vwxyzjn/cleanrl), config a new env for the cloned repo. **Please check the [official repo](https://github.com/vwxyzjn/cleanrl) for detailed instruction.**  
```bash
conda create -n clean-rl python=3.8
conda activate clean-rl
pip install -r requirements/requirements.txt  # for cartpole only
python cleanrl/ppo.py --seed 1 --env-id CartPole-v0 --total-timesteps 50000 --capture_video  # cd ./videos for visualized results
tensorboard --logdir runs  # open another terminal to see the training process
```

## Usage

Train your model using
```bash
python ppo_continuous.py --total-timesteps 1000000  # add necessary arguments for example
tensorboard --logdir runs  # enable visualization in another terminal
```
Please modify arguments in ppo_continuous.py and f110_env_rl.py for expected config. 

Test your model using
```bash
python inference.py
```
Please modify arguments in inference.py to match the validation setup. 

Please check the youtube video [playlist](https://www.youtube.com/playlist?list=PLxQD__8MmsvBaEOhgLCPge5m_NwwU4zfs) for further details. 