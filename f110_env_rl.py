# -- coding: utf-8 --

"""
    F1TENTH gym environment of the RL planner
    Author: Derek Zhou, Biao Wang
"""

import os

import numpy as np
import yaml

from controllers.pure_pursuit import PurePursuit, get_lookahead_point
from controllers.kmpc import MPCConfig_F110, KinematicModel, KMPCController
from f110_gym.envs.base_classes import RaceCar
from f110_gym.envs.f110_env import F110Env
from gym import spaces
from utils.lidar_utils import downsample_lidar_scan
from utils.render import Renderer, fix_gui
from utils.traj_utils import get_front_traj, get_interpolated_traj_with_horizon, densify_offset_traj, get_offset_traj, \
    global_to_local, local_to_global, bresenham_line_index
from utils.waypoint_loader import WaypointLoader, waypoints_dir_correction


class F110RLEnv(F110Env):
    def __init__(self, **kwargs):
        self.usage = 'nudge'  # !!!! bt or nudge

        # load keyword arguments
        self.render_flag = kwargs['render']
        map_name = kwargs['map_name']  # levine_2nd, skir
        self.num_obstacles = kwargs['num_obstacles']
        self.num_lidar_scan = kwargs['num_lidar_scan']
        self.ctrl_method = kwargs['ctrl_method']

        # load map, waypoints, controller, and renderer
        map_path = os.path.abspath(os.path.join('maps', map_name))
        csv_data = np.loadtxt(map_path + '/' + map_name + '_raceline.csv', delimiter=';', skiprows=0)

        with open(map_path + '/' + map_name + '_map' + '.yaml', 'r') as yaml_stream:
            try:
                map_metadata = yaml.safe_load(yaml_stream)
                self.map_resolution = map_metadata['resolution']
                self.origin = map_metadata['origin']
            except yaml.YAMLError as ex:
                print(ex)

        self.orig_x = self.origin[0]
        self.orig_y = self.origin[1]

        self.waypoints = WaypointLoader(map_name, csv_data)
        if self.ctrl_method == 'pure_pursuit':
            self.lookahead_dist = 0.8  # !!!!
            self.controller = PurePursuit(self.waypoints, self.lookahead_dist)
        elif self.ctrl_method == 'kinematic_mpc':
            model_config = MPCConfig_F110()
            kmpc_waypoints = waypoints_dir_correction(map_name, csv_data)
            self.controller = KMPCController(model=KinematicModel(config=model_config), waypoints=kmpc_waypoints,
                                             config=model_config)
            self.control_period = 10  # Hz
        self.renderer = Renderer(self.waypoints)
        if self.render_flag:
            super().add_render_callback(self.renderer.render_waypoints)
            # super().add_render_callback(self.renderer.render_front_traj)
            super().add_render_callback(self.renderer.render_horizon_traj)
            if self.ctrl_method == 'pure_pursuit':
                super().add_render_callback(self.renderer.render_lookahead_point)
            super().add_render_callback(self.renderer.render_offset_traj)
            super().add_render_callback(fix_gui)

        if 'obt_poses' in kwargs:
            # get the obstacle poses input
            obt_pose = kwargs['obt_poses']
        else:
            # randomly generate obstacles
            obt_index = np.random.uniform(1, self.waypoints.x.shape[0] - 1, size=(self.num_obstacles,)).astype(int)
            thetas = np.arctan2(self.waypoints.x[obt_index + 1] - self.waypoints.x[obt_index - 1],
                                self.waypoints.y[obt_index + 1] - self.waypoints.y[obt_index - 1])
            obt_pose = np.array([self.waypoints.x[obt_index],
                                 self.waypoints.y[obt_index], thetas]).transpose().reshape((-1, 3))

        # load the super class - F110Env
        super(F110RLEnv, self).__init__(map=map_path + '/' + map_name + '_map',
                                        map_ext='.pgm' if map_name == 'levine_2nd' or map_name == 'skir' or map_name == 'skir_blocked' else '.png',
                                        seed=0, num_agents=1, obt_poses=obt_pose)

        # init params
        self.horizon = int(10)
        self.predict_time = 2.0  # if self.ctrl_method == 'kinematic_mpc' else 2.0 or 1.0   # !!!!
        self.fixed_speed = 2.0
        self.offset = [0.5] * self.horizon
        self.steering = 0.0
        self.speed = 0.0

        self.map_max_rows = RaceCar.scan_simulator.map_img.shape[0]
        self.map_max_cols = RaceCar.scan_simulator.map_img.shape[1]

        # set up the bounding boxes
        self.max_lidar_range = 30
        self.max_pose = 1e3
        self.min_pose = -self.max_pose
        self.max_offset = 1
        self.min_offset = -self.max_offset

        low_lidar = 0 * np.ones((self.num_lidar_scan,), dtype=np.float32)
        high_lidar = self.max_lidar_range * np.ones((self.num_lidar_scan,), dtype=np.float32)
        low_traj = self.min_pose * np.ones((self.horizon * 2,), dtype=np.float32)
        high_traj = self.max_pose * np.ones((self.horizon * 2,), dtype=np.float32)
        low_pose = self.min_pose * np.ones((2,), dtype=np.float32)
        high_pose = self.max_pose * np.ones((2,), dtype=np.float32)
        obs_low_bound = np.hstack((low_lidar, low_traj, low_pose))
        obs_high_bound = np.hstack((high_lidar, high_traj, high_pose))
        self.observation_space = spaces.Box(low=obs_low_bound, high=obs_high_bound,
                                            shape=(obs_high_bound.shape[0],), dtype=np.float32)
        self.single_observation_space = spaces.Box(low=obs_low_bound, high=obs_high_bound,
                                                   shape=(obs_high_bound.shape[0],), dtype=np.float32)
        # print("observation space shape", self.single_observation_space.shape)

        self.action_space = spaces.Box(low=self.min_offset, high=self.max_offset,
                                       shape=(self.horizon,), dtype=np.float32)  # action: offsets in n horizons
        self.single_action_space = spaces.Box(low=self.min_offset, high=self.max_offset,
                                              shape=(self.horizon,), dtype=np.float32)
        # print("action space shape", self.single_action_space.shape)

    def get_network_obs(self):
        lidar_obs = downsample_lidar_scan(self.obs['scans'][0].flatten(), self.num_lidar_scan)
        traj_obs = self.local_horizon_traj[:, :2].flatten()
        pose_obs = np.array([self.obs['poses_x'][0], self.obs['poses_y'][0]]).reshape((-1,))
        network_obs = np.hstack((lidar_obs, traj_obs, pose_obs))

        return network_obs

    def reset(self, seed=1):
        # initialization
        init_index = np.random.randint(0, self.waypoints.x.shape[0])
        init_pos = np.array([self.waypoints.x[init_index], self.waypoints.y[init_index],
                             self.waypoints.θ[init_index]]).reshape((1, -1))
        # init_pos = np.array([[0.0, 0.0, 0.0]])  # fixed init or not

        self.obs, _, self.done, _ = super().reset(init_pos)  # self.obs, _, self.done, _ = F110Env.reset(self,init_pos)
        self.lap_time = 0.0

        # get init horizon traj
        self.front_traj = get_front_traj(self.obs, self.waypoints, predict_time=self.predict_time)  # [i, x, y, v, θ]
        self.horizon_traj = get_interpolated_traj_with_horizon(self.front_traj, self.horizon)  # [x, y, v, θ]
        self.local_horizon_traj = global_to_local(self.obs, self.horizon_traj)  # [x, y, v, θ]
        self.local_offset_traj = get_offset_traj(self.local_horizon_traj, self.offset)
        self.offset_traj = local_to_global(self.obs, self.local_offset_traj)
        self.offset_traj = np.vstack((np.array([[self.obs['poses_x'][0], self.obs['poses_y'][0],
                                                 self.fixed_speed, self.obs['poses_theta'][0]]]), self.offset_traj))

        if self.render_flag:
            # self.renderer.front_traj = self.front_traj
            self.renderer.horizon_traj = self.horizon_traj
            self.renderer.offset_traj = self.offset_traj

        network_obs = self.get_network_obs()
        return network_obs

    def step(self, offset=None):
        self.offset = offset
        # add offsets on horizon traj & densify offset traj to 80 points & get lookahead point & pure pursuit
        self.local_offset_traj = get_offset_traj(self.local_horizon_traj, self.offset)
        self.offset_traj = local_to_global(self.obs, self.local_offset_traj)
        if self.ctrl_method == 'pure_pursuit':
            self.offset_traj = np.vstack((np.array([[self.obs['poses_x'][0], self.obs['poses_y'][0],
                                                     self.fixed_speed, self.obs['poses_theta'][0]]]), self.offset_traj))
            if self.usage == 'nudge':
                dense_offset_traj = densify_offset_traj(self.offset_traj)  # [x, y, v, theta] for obstacle nudging
            elif self.usage == 'bt':
                # for bootstrap only! -> Behavioral Cloning
                dense_offset_traj = densify_offset_traj(self.horizon_traj)
            lookahead_point_profile = get_lookahead_point(self.obs, dense_offset_traj,
                                                          lookahead_dist=self.lookahead_dist)
            self.steering, self.speed = self.controller.rl_control(self.obs, lookahead_point_profile,
                                                                   max_speed=self.fixed_speed)
        elif self.ctrl_method == 'kinematic_mpc':
            mpc_offset_traj = densify_offset_traj(self.offset_traj, intep_num=11)
            if int(self.lap_time * 100) % self.control_period == 0:  # 50 ms
                veh_state = np.array([self.sim.agents[0].state[0],
                                      self.sim.agents[0].state[1],
                                      self.sim.agents[0].state[3],  # vx
                                      self.sim.agents[0].state[4],  # yaw angle
                                      ])
                self.steering, self.speed, ref_path_x, ref_path_y, pred_x, pred_y, mpc_ox, mpc_oy, a = self.controller.rl_control(
                    veh_state, mpc_offset_traj)
                # renderer.offset_traj = np.array([ref_path_x, ref_path_y]).T  # red
                # renderer.horizon_traj = np.array([pred_x, pred_y]).T  # yellow

        # step function in race car, time step is k+1 now
        # print("steering = {}, speed = {}".format(round(self.steering, 4), round(self.speed, 4)))
        self.obs, step_time, self.done, info = super().step(
            np.array([[self.steering, self.speed]]))  # not fixed for mpc
        self.lap_time += step_time

        # extract waypoints in predicted time & interpolate the front traj to get a 10-point-traj
        self.front_traj = get_front_traj(self.obs, self.waypoints, predict_time=self.predict_time)  # [i, x, y, v, θ]
        self.horizon_traj = get_interpolated_traj_with_horizon(self.front_traj, self.horizon)  # [x, y, v, θ]
        self.local_horizon_traj = global_to_local(self.obs, self.horizon_traj)  # [x, y, v, θ]

        # get agent observation [lidar, front traj, pose]
        network_obs = self.get_network_obs()

        # offset trajectory collision predictions
        offset_x_index = np.ceil((self.offset_traj[:, 0] - self.orig_x) / self.map_resolution).astype(int)
        offset_y_index = np.ceil((self.offset_traj[:, 1] - self.orig_y) / self.map_resolution).astype(int)
        offset_traj_indices = np.vstack((offset_x_index, offset_y_index)).T
        all_indices = []
        for i in range(offset_traj_indices.shape[0] - 1):
            line_indices = bresenham_line_index(offset_traj_indices[i, :], offset_traj_indices[i + 1, :])
            all_indices.append(line_indices)
        all_indices = np.concatenate(all_indices).reshape(-1, 2)
        filtered_traj_indices = all_indices[
            (all_indices[:, 1] < self.map_max_rows) & (all_indices[:, 0] < self.map_max_cols) &
            (all_indices[:, 1] >= 0) & (all_indices[:, 0] >= 0)]

        # modify your reward
        # derek's reward for bootstrapping
        if self.usage == 'bt':
            reward = 100 * step_time
            reward -= 1 * np.linalg.norm(offset, ord=2)
            if super().current_obs['collisions'][0] == 1:
                reward -= 1000

        # modify your reward
        # !!!! derek's reward for obstacle avoidance
        if self.usage == 'nudge':
            reward = 100 * step_time
            # reward -= 0.1 * np.linalg.norm(offset, ord=2)
            first_diff = (offset[1:] - offset[:-1])
            second_diff = first_diff[1:] - first_diff[:-1]
            reward -= 0.2 * np.linalg.norm(first_diff, ord=2)  # < 0.2
            reward -= 0.1 * np.linalg.norm(second_diff, ord=2)  # < 0.2
            reward -= 0.05 * np.count_nonzero(
                RaceCar.scan_simulator.map_img[filtered_traj_indices[:, 1], filtered_traj_indices[:, 0]] == 0)  # < 0.5
            if super().current_obs['collisions'][0] == 1:
                reward -= 1000

        if self.render_flag:  # render update
            self.renderer.offset_traj = self.offset_traj
            if self.ctrl_method == 'pure_pursuit':
                self.renderer.ahead_point = lookahead_point_profile[:2]  # [x, y]
            # self.renderer.front_traj = self.front_traj
            self.renderer.horizon_traj = self.horizon_traj
            super().render('human')

        return network_obs, reward, self.done, info
