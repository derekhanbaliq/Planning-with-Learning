# ESE-650-Final-Project

TODO: Project Intro

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

Test your environment.
```bash
python test.py  # check if cuda is available
cd examples
python waypoint_follow.py  # check f1tenth_gym is working
```

To run [cleanrl](https://github.com/vwxyzjn/cleanrl) for studying, config a new env for the cloned repo. **Please check the repo for detailed instruction.**  
```bash
conda create -n clean-rl python=3.8
conda activate clean-rl
pip install -r requirements/requirements.txt  # for cartpole only
python cleanrl/ppo.py --seed 1 --env-id CartPole-v0 --total-timesteps 50000 --capture_video  # cd ./videos for visualized results
tensorboard --logdir runs  # open another terminal to see the training process
```

## Usage

TBD