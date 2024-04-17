# -- coding: utf-8 --

"""
    F1TENTH gym environment of the RL planner
    Author: Derek Zhou, Biao Wang, Tian Tan
"""

import os

import numpy as np

from controllers.pure_pursuit import PurePursuit, get_lookahead_point
from f110_gym.envs.f110_env import F110Env
from gym import spaces
from utils.render import Renderer, fix_gui
from utils.traj_utils import get_front_traj, get_interpolated_traj_with_horizon, densify_offset_traj, get_offset_traj
from utils.waypoint_loader import WaypointLoader


class F110RLEnv(F110Env):
    def __init__(self, **kwargs):
        # render flag
        self.render_flag = kwargs['render']

        # load map
        map_name = 'levine_2nd'  # levine_2nd, skir
        map_path = os.path.abspath(os.path.join('maps', map_name))

        # load waypoints
        csv_data = np.loadtxt(map_path + '/' + map_name + '_raceline.csv', delimiter=';', skiprows=0)
        self.waypoints = WaypointLoader(map_name, csv_data)

        # load controller
        self.controller = PurePursuit(self.waypoints)

        # load renderer
        self.renderer = Renderer(self.waypoints)
        if self.render_flag:
            super().add_render_callback(self.renderer.render_waypoints)
            super().add_render_callback(self.renderer.render_front_traj)
            super().add_render_callback(self.renderer.render_horizon_traj)
            super().add_render_callback(self.renderer.render_lookahead_point)
            super().add_render_callback(self.renderer.render_offset_traj)
            super().add_render_callback(fix_gui)

        # load F110Env
        super(F110RLEnv, self).__init__(map=map_path + '/' + map_name + '_map',
                                        map_ext='.pgm' if map_name == 'levine_2nd' or map_name == 'skir' else '.png',
                                        seed=0, num_agents=1)

        # init params
        self.horizon = int(10)
        self.predict_time = 1.0  # get waypoints in coming seconds
        self.rl_max_speed = 5.0
        self.offset = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]

        # TODO: find a better way to config these params
        # self.num_beam = self.f110_env.sim.agents[0].num_beams
        # self.max_lidar_range = self.f110_env.sim.agents[0].scan_simulator.max_range

        self.num_beam = 1080
        self.max_lidar_range = 30

        self.max_pose = 1e3
        self.min_pose = -self.max_pose

        self.max_offset = 1
        self.min_offset = -self.max_offset

        low_lidar = 0 * np.ones((self.num_beam,), dtype=np.float32)
        high_lidar = self.max_lidar_range * np.ones((self.num_beam,), dtype=np.float32)

        low_traj = self.min_pose * np.ones((self.horizon * 2,), dtype=np.float32)
        high_traj = self.max_pose * np.ones((self.horizon * 2,), dtype=np.float32)

        low_pose = self.min_pose * np.ones((2,), dtype=np.float32)
        high_pose = self.max_pose * np.ones((2,), dtype=np.float32)

        obs_low_bound = np.hstack((low_lidar, low_traj, low_pose))
        obs_high_bound = np.hstack((high_lidar, high_traj, high_pose))

        self.observation_space = spaces.Box(low=obs_low_bound, high=obs_high_bound, shape=(obs_high_bound.shape[0],),
                                            dtype=np.float32)
        self.single_observation_space = spaces.Box(low=obs_low_bound, high=obs_high_bound,
                                                   shape=(obs_high_bound.shape[0],), dtype=np.float32)
        # action: offsets in n horizons
        self.action_space = spaces.Box(low=self.min_offset, high=self.max_offset, shape=(self.horizon,),
                                       dtype=np.float32)
        self.single_action_space = spaces.Box(low=self.min_offset, high=self.max_offset, shape=(self.horizon,),
                                              dtype=np.float32)
        # print("observation space shape", self.single_observation_space.shape)
        # print("action space shape", self.single_action_space.shape)

    def get_network_obs(self):
        lidar_obs = self.obs['scans'][0].flatten()
        traj_obs = self.horizon_traj[:, :2].flatten()
        pose_x_obs = self.obs['poses_x'][0]
        pose_y_obs = self.obs['poses_y'][0]
        pose_obs = np.array([pose_x_obs, pose_y_obs]).reshape((-1,))
        network_obs = np.hstack((lidar_obs, traj_obs, pose_obs))

        return network_obs

    def reset(self, seed=1):
        # initialization
        init_pos = np.array([0.0, 0.0, 0.0]).reshape((1, -1))  # 1 x 3
        self.obs, _, self.done, _ = super().reset(init_pos)
        # self.obs, _, self.done, _ = F110Env.reset(self,init_pos)
        self.lap_time = 0.0

        # get init horizon traj
        self.front_traj = get_front_traj(self.obs, self.waypoints, predict_time=self.predict_time)  # [i, x, y, v]
        self.horizon_traj = get_interpolated_traj_with_horizon(self.front_traj, self.horizon)  # [x, y, v]
        self.offset_traj = get_offset_traj(self.horizon_traj, self.offset)

        if self.render_flag:
            self.renderer.front_traj = self.front_traj
            self.renderer.horizon_traj = self.horizon_traj
            self.renderer.render_offset_traj = self.offset_traj

        network_obs = self.get_network_obs()
        return network_obs

    def step(self, offset=None):
        # offset = [0., 0.1, 0.2, 0.3, 0.4, 0.4, 0.3, 0.2, 0.1, 0.0]  # fake offset, [-1, 1], half width [right, left]
        self.offset = offset
        # add offsets on horizon traj & densify offset traj to 80 points & get lookahead point & pure pursuit
        self.offset_traj = get_offset_traj(self.horizon_traj, self.offset)
        dense_offset_traj = densify_offset_traj(self.horizon_traj)  # [x, y, v]
        lookahead_point_profile = get_lookahead_point(dense_offset_traj, lookahead_dist=1.5)
        steering, speed = self.controller.rl_control(self.obs, lookahead_point_profile, max_speed=self.rl_max_speed)

        # step function in race car, time step is k+1 now
        self.obs, step_time, raw_done, raw_info = super().step(np.array([[steering, speed]]))
        self.lap_time += step_time

        # extract waypoints in predicted time & interpolate the front traj to get a 10-point-traj
        self.front_traj = get_front_traj(self.obs, self.waypoints, predict_time=self.predict_time)  # [i, x, y, v]
        self.horizon_traj = get_interpolated_traj_with_horizon(self.front_traj, self.horizon)  # [x, y, v]

        # TODO: modify the next observation output (lidar, front traj, pose)
        network_obs = self.get_network_obs()

        # TODO: design the reward function
        reward = 10 * step_time
        reward -= 1 * np.linalg.norm(offset, ord=2)
        
        if super().current_obs['collisions'][0] == 1:
            reward -= 100

        self.done = raw_done
        info = raw_info

        if self.render_flag:  # render update
            self.renderer.offset_traj = self.offset_traj
            self.renderer.ahead_point = lookahead_point_profile[:2]  # [x, y]
            self.renderer.front_traj = self.front_traj
            self.renderer.horizon_traj = self.horizon_traj
            super().render('human')

        return network_obs, reward, self.done, info

    # def reset(self, init_pose):
    #     obs, reward, done, info = super().reset(init_pose)
    #     return obs, reward, done, info

    # def render(self, mode='human'):
    #     super().render(mode)
