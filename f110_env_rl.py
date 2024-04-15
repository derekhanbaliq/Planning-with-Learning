# -- coding: utf-8 --
import os

import numpy as np

from controllers.pure_pursuit import PurePursuit, get_lookahead_point
from f110_gym.envs.f110_env import F110Env
from gym import spaces
from utils.render import Renderer, fix_gui
from utils.rl_utils import get_front_traj, get_interpolated_traj_with_horizon, densify_offset_traj, get_offset_traj
from utils.waypoint_loader import WaypointLoader


class F110RLEnv(F110Env):
    def __init__(self, **kwargs):
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

        # initialization
        init_pos = np.array([0.0, 0.0, 0.0]).reshape((1, -1))  # 1 x 3
        self.obs, _, self.done, _ = super().reset(init_pos)
        self.lap_time = 0.0

        # get init horizon traj
        self.front_traj = get_front_traj(self.obs, self.waypoints, predict_time=self.predict_time)  # [i, x, y, v]
        self.renderer.front_traj = self.front_traj
        self.horizon_traj = get_interpolated_traj_with_horizon(self.front_traj, self.horizon)  # [x, y, v]
        self.renderer.horizon_traj = self.horizon_traj
        self.offset_traj = get_offset_traj(self.horizon_traj,self.offset)
        self.renderer.render_offset_traj = self.offset_traj

        # TODO: find a better way to config these params
        # self.num_beam = self.f110_env.sim.agents[0].num_beams
        # self.max_lidar_range = self.f110_env.sim.agents[0].scan_simulator.max_range

        self.num_beam = 1080
        self.max_lidar_range = 30

        self.max_pose = 1e3
        self.min_pose = -self.max_pose

        self.max_offset = 10
        self.min_offset = -self.max_offset

        # observation: lidar scans, reference front points, current x&y positions
        # need to be changed to Box
        self.single_observation_space = spaces.Dict(
            {"scan": spaces.Box(low=0, high=self.max_lidar_range, shape=(self.num_beam,), dtype=np.float32),
             "front": spaces.Box(low=self.min_pose, high=self.max_pose, shape=(self.horizon,),
                                 dtype=np.float32),
             "pose": spaces.Box(low=self.min_pose, high=self.max_pose, shape=(2,), dtype=np.float32)})

        # action: offsets in n horizons
        self.single_action_space = spaces.Box(low=self.min_offset, high=self.max_offset, shape=(self.horizon,),
                                              dtype=np.float32)

        print("observation space shape", self.single_observation_space.shape)
        print("action space shape", self.single_action_space.shape)

    def env_step(self, offset=None):
        # offset = [0., 0.1, 0.2, 0.3, 0.4, 0.4, 0.3, 0.2, 0.1, 0.0]  # fake offset, [-1, 1], half width [right, left]

        # add offsets on horizon traj & densify offset traj to 80 points & get lookahead point & pure pursuit
        self.offset_traj = get_offset_traj(self.horizon_traj, self.offset)
        self.renderer.offset_traj = self.offset_traj
        dense_offset_traj = densify_offset_traj(self.horizon_traj)  # [x, y, v]
        lookahead_point_profile = get_lookahead_point(dense_offset_traj, lookahead_dist=1.5)
        self.renderer.ahead_point = lookahead_point_profile[:2]  # [x, y]
        steering, speed = self.controller.rl_control(self.obs, lookahead_point_profile, max_speed=self.rl_max_speed)
        # print("steering = {}, speed = {}".format(round(steering, 5), round(speed, 5)))

        # step function in race car, time step is k+1 now
        self.obs, step_time, raw_done, raw_info = super().step(np.array([[steering, speed]]))
        self.lap_time += step_time

        # extract waypoints in predicted time & interpolate the front traj to get a 10-point-traj
        self.front_traj = get_front_traj(self.obs, self.waypoints, predict_time=self.predict_time)  # [i, x, y, v]
        self.renderer.front_traj = self.front_traj
        self.horizon_traj = get_interpolated_traj_with_horizon(self.front_traj, self.horizon)  # [x, y, v]
        self.renderer.horizon_traj = self.horizon_traj

        # TODO: design the reward function

        # TODO: modify the next observation output (lidar, front traj, pose)
        # only have horizon traj
        observation = self.horizon_traj.flatten()  # + lidar & pose

        # observation = raw_obs
        reward = step_time
        self.done = raw_done
        info = raw_info

        super().render('human')

        return observation, reward, self.done, info

    # def reset(self, init_pose):
    #     obs, reward, done, info = super().reset(init_pose)
    #     return obs, reward, done, info

    # def render(self, mode='human'):
    #     super().render(mode)
