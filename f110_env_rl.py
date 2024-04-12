# -- coding: utf-8 --
import os
import gym
from gym import spaces
import numpy as np

from f110_gym.envs.f110_env import F110Env

import yaml

from controllers.pure_pursuit import PurePursuit, get_lookahead_point
# from f110_env_rl import F110RLEnv
from utils.render import Renderer, fix_gui
from utils.rl_utils import get_front_traj, get_interpolated_traj_with_horizon, densify_offset_traj
from utils.waypoint_loader import WaypointLoader



class F110RLEnv(F110Env):
    def __init__(self, **kwargs):
        
        # load map & yaml
        map_name = 'skir'  # levine_2nd, skir, Spielberg, MoscowRaceway, Catalunya
        map_path = os.path.abspath(os.path.join('maps', map_name))
        # yaml_config = yaml.load(open(map_path + '/' + map_name + '_map.yaml'), Loader=yaml.FullLoader)

        # load waypoints
        csv_data = np.loadtxt(map_path + '/' + map_name + '_raceline.csv', delimiter=';', skiprows=0)  # '_centerline.csv' '_raceline.csv'
        self.waypoints = WaypointLoader(map_name, csv_data)

        # load controller
        self.controller = PurePursuit(self.waypoints)
        
        self.renderer = Renderer(self.waypoints)
        super().add_render_callback(self.renderer.render_waypoints)
        super().add_render_callback(self.renderer.render_front_traj)
        super().add_render_callback(self.renderer.render_horizon_traj)
        super().add_render_callback(self.renderer.render_lookahead_point)
        super().add_render_callback(fix_gui)
        
        super(F110RLEnv, self).__init__(seed=0, map=map_path + '/' + map_name + '_map',
                    map_ext='.pgm' if map_name == 'levine_2nd' or map_name == 'skir' else '.png', num_agents=1)

        self.rl_max_speed = 5.0
        self.horizon = int(10)
        self.predict_time = 1.0
        
        self.lap_time = 0.0

        # initialization
        init_pos = np.array([0.0, 0.0, 0.0]).reshape((1, -1))
        self.obs, _, self.done, _ = super().reset(init_pos)

        self.front_traj = get_front_traj(self.obs, self.waypoints, predict_time=self.predict_time)  # [i, x, y, v]
        self.renderer.front_traj = self.front_traj
        # print(front_traj.shape)

        # interpolate the front traj to get a 10-point-traj
        self.horizon_traj = get_interpolated_traj_with_horizon(self.front_traj, self.horizon)  # [x, y, v]
        self.renderer.horizon_traj = self.horizon_traj


        
        # waypoint, controller, render

        # self.f110_env = f110

        # # TODO: find a better way to config these params
        # self.num_beam = self.f110_env.sim.agents[0].num_beams
        # self.max_lidar_range = self.f110_env.sim.agents[0].scan_simulator.max_range
        
        self.num_beam = 1080
        self.max_lidar_range = 30
        
        self.num_front_point = 10
        
        self.max_pose = 1e3
        self.min_pose = -self.max_pose
        
        self.max_offset = 10
        self.min_offset = -self.max_offset
        
        # observation: lidar scans, reference front points, current x&y positions
        # need to be changed to Box
        self.single_observation_space = spaces.Dict(
            {"scan": spaces.Box(low=0, high=self.max_lidar_range, shape=(self.num_beam,), dtype=np.float32),
             "front": spaces.Box(low=self.min_pose, high=self.max_pose, shape=(self.num_front_point,), dtype=np.float32),
             "pose": spaces.Box(low=self.min_pose, high=self.max_pose, shape=(2,), dtype=np.float32)})
        
        # action: offsets in n horizons
        self.single_action_space = spaces.Box(low=self.min_offset, high=self.max_offset, shape=(self.horizon,), dtype=np.float32)
        
        print("observation space shape", self.single_observation_space.shape)
        print("action space shape", self.single_action_space.shape)

    def env_step(self, offset=None):
        # TODO: lidar scan & h-traj -> PPO -> lateral offsets
        # offset = PPO_model(obs['scan'], horizon_traj)
        # offset = [0., 0.1, 0.2, 0.3, 0.4, 0.4, 0.3, 0.2, 0.1, 0.0]  # fake offset, [-1, 1], half width [right, left]

        # TODO: interpolate the offsets for every waypoint in traj -> get offset traj in frenet frame
        # TODO: transform it into world frame
        # offset_horizon_traj =   # len = 10
        # TODO: transform the action (from offset to real input)

        
        dense_offset_traj = densify_offset_traj(self.horizon_traj)  # [x, y, v]
        lookahead_point_profile = get_lookahead_point(dense_offset_traj, lookahead_dist=1.5)
        self.renderer.ahead_point = lookahead_point_profile[:2]  # [x, y]
        
        steering, speed = self.controller.rl_control(self.obs, lookahead_point_profile, max_speed=self.rl_max_speed)
        # print("steering = {}, speed = {}".format(round(steering, 5), round(speed, 5)))
        
        
        
        
        # step function in race car
        # obs in time k+1
        self.obs, step_time, raw_done, raw_info = super().step(np.array([[steering, speed]]))
        self.lap_time += step_time

        self.front_traj = get_front_traj(self.obs, self.waypoints, predict_time=self.predict_time)  # [i, x, y, v]
        self.renderer.front_traj = self.front_traj
        # print(front_traj.shape)

        # interpolate the front traj to get a 10-point-traj
        self.horizon_traj = get_interpolated_traj_with_horizon(self.front_traj, self.horizon)  # [x, y, v]
        self.renderer.horizon_traj = self.horizon_traj
        # TODO: need to simulate for n horizon/time steps

        # TODO: design the reward function

        # only have horizon traj
        observation = self.horizon_traj.flatten() # + lidar & pose

        # observation = raw_obs
        reward = step_time
        done = raw_done
        info = raw_info

        super().render('human')

        return observation, reward, done, info

    # def reset(self, init_pose):
    #     obs, reward, done, info = super().reset(init_pose)
    #     return obs, reward, done, info

    # def render(self, mode='human'):
    #     super().render(mode)
