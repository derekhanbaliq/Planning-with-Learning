"""
    main application for RL planner
    Author: Derek Zhou, Biao Wang, Tian Tan
    References: https://f1tenth-gym.readthedocs.io/en/v1.0.0/api/obv.html
                https://github.com/f1tenth/f1tenth_gym/tree/main/examples
"""

import os

import numpy as np
import yaml

from controllers.lqr_steering import LQRSteeringController
from controllers.lqr_steering_speed import LQRSteeringSpeedController
from controllers.pure_pursuit import PurePursuit, get_lookahead_point
from f110_gym.envs.f110_env import F110Env
from utils.render import Renderer, fix_gui
from utils.rl_utils import get_front_traj, get_interpolated_traj_with_horizon, densify_offset_traj, get_offset_traj
from utils.waypoint_loader import WaypointLoader
from f110_env_rl import F110RLEnv

from ppo_continuous import Agent
import torch


def main():
    method = 'pure_pursuit'  # pure_pursuit, lqr_steering, lqr_steering_speed
    rl_planner = True

    # load map & yaml
    map_name = 'levine_2nd'  # levine_2nd, skir, Spielberg, MoscowRaceway, Catalunya
    map_path = os.path.abspath(os.path.join('maps', map_name))
    yaml_config = yaml.load(open(map_path + '/' + map_name + '_map.yaml'), Loader=yaml.FullLoader)

    # load waypoints
    csv_data = np.loadtxt(map_path + '/' + map_name + '_raceline.csv', delimiter=';', skiprows=0)  # '_centerline.csv'
    waypoints = WaypointLoader(map_name, csv_data)

    # load controller
    controller = None
    if method == 'pure_pursuit':
        controller = PurePursuit(waypoints)
    elif method == 'lqr_steering':
        controller = LQRSteeringController(waypoints)
    elif method == 'lqr_steering_speed':
        controller = LQRSteeringSpeedController(waypoints)

    # create & init env
    # env = gym.make('f110_gym:f110-v0', map=map_path + '/' + map_name + '_map',
    #                map_ext='.pgm' if map_name == 'levine_2nd' or map_name == 'skir' else '.png', num_agents=1)
    env = F110Env(map=map_path + '/' + map_name + '_map',
                  map_ext='.pgm' if map_name == 'levine_2nd' or map_name == 'skir' else '.png', num_agents=1)

    renderer = Renderer(waypoints)
    env.add_render_callback(renderer.render_waypoints)
    env.add_render_callback(renderer.render_front_traj) if rl_planner else None
    env.add_render_callback(renderer.render_horizon_traj) if rl_planner else None
    env.add_render_callback(renderer.render_lookahead_point) if rl_planner else None
    env.add_render_callback(renderer.render_offset_traj) if rl_planner else None
    env.add_render_callback(fix_gui)
    lap_time = 0.0
    init_pos = np.array([yaml_config['init_pos']])
    obs, _, done, _ = env.reset(init_pos)

    horizon = int(10)
    rl_max_speed = 5.0

    rl_env = F110RLEnv(render=False)
    model = Agent(rl_env)
    model.load_state_dict(torch.load('test.pkl'))

    while not done:
        if method == 'pure_pursuit' and rl_planner:
            # extract waypoints in coming seconds as front traj
            front_traj = get_front_traj(obs, waypoints, predict_time=1.0)  # [i, x, y, v]
            renderer.front_traj = front_traj
            # print(front_traj.shape)

            # interpolate the front traj to get a 10-point-traj
            horizon_traj = get_interpolated_traj_with_horizon(front_traj, horizon)  # [x, y, v]
            renderer.horizon_traj = horizon_traj

            # TODO: refine the code please!
            def get_network_obs(obs, horizon_traj):
                lidar_obs = obs['scans'][0].flatten()
                traj_obs = horizon_traj[:, :2].flatten()
                pose_x_obs = obs['poses_x'][0]
                pose_y_obs = obs['poses_y'][0]
                pose_obs = np.array([pose_x_obs, pose_y_obs]).reshape((-1,))
                network_obs = np.hstack((lidar_obs, traj_obs, pose_obs))
                return network_obs

            network_obs = get_network_obs(obs, horizon_traj)
            # print(network_obs.shape)
            network_obs = torch.from_numpy(network_obs).to(dtype=torch.float).resize(1, network_obs.shape[0])
            action, sum_log_prob, sum_entropy, value = model.get_action_and_value(network_obs)
            offset = action.numpy().flatten()
            print(offset)

            offset_traj = get_offset_traj(horizon_traj, offset)
            renderer.offset_traj = offset_traj

            # interpolate offset traj to get the lookahead point profile (x, y) and v
            dense_offset_traj = densify_offset_traj(offset_traj)  # [x, y, v]
            lookahead_point_profile = get_lookahead_point(dense_offset_traj, lookahead_dist=1.5)
            renderer.ahead_point = lookahead_point_profile[:2]  # [x, y]

            # input lookahead point pos & ref speed, output steering & speed
            steering, speed = controller.rl_control(obs, lookahead_point_profile, max_speed=rl_max_speed)

        else:
            steering, speed = controller.control(obs)

        print("steering = {}, speed = {}".format(round(steering, 5), round(speed, 5)))

        obs, time_step, done, _ = env.step(np.array([[steering, speed]]))

        lap_time += time_step
        env.render(mode='human')

    print('Sim elapsed time:', lap_time)


if __name__ == '__main__':
    main()
