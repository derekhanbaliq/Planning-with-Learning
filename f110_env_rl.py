# -- coding: utf-8 --
import gym
from gym import spaces
import numpy as np

from f110_gym.envs.f110_env import F110Env


class F110Env_Continuous_Planner(gym.Env):
    def __init__(self, f110:F110Env, horizon=1):
        super(F110Env_Continuous_Planner, self).__init__()
        
        self.f110_env = f110
        
        # TODO: find a better way to config these params
        self.num_beam = self.f110_env.sim.agents[0].num_beams
        self.max_lidar_range = self.f110_env.sim.agents[0].scan_simulator.max_range
        
        self.num_front_point = 10
        
        self.max_pose = 1e3
        self.min_pose = -self.max_pose

        self.max_offset = 10
        self.min_offset = -self.max_offset
        
        # observation: lidar scans, reference front points, current x&y positions
        self.observation_space = spaces.Dict({"scan": spaces.Box(low=0, high=self.max_lidar_range, shape=(self.num_beam,), dtype=np.float32), 
                                            "front": spaces.Box(low=self.min_pose, high=self.max_pose, shape=(self.num_front_point), dtype=np.float32),
                                            "pose": spaces.Box(low=self.min_pose, high=self.max_pose, shape=(2,), dtype=np.float32)})
        
        # action: offsets in n horizons
        self.action_space = spaces.Box(low=self.min_offset, high=self.max_offset, shape=(horizon,), dtype=np.float32)
        
    
    def step(self, action):
        # TODO: transform the action (from offset to real input)
        raw_obs, raw_reward, raw_done, raw_info = self.f110_env.step(action)
        
        # TODO: need to simulate for n horizon/time steps
        
        # TODO: design the reward function
        
        observation = raw_obs
        reward = raw_reward
        done = raw_done
        info = raw_info        
        
        return observation, reward, done, info
    
    
    def reset(self, init_pose):
        obs, _, _, info =  self.f110_env.reset(init_pose)
        return obs, info
    
    
    def render(self, mode='human'):
        self.f110_env.render(mode)
        
        
    def close(self):
        return super().close()
        
        