"""
    main application for RL planner
    Author: Derek Zhou, Biao Wang, Tian Tan
    References: https://f1tenth-gym.readthedocs.io/en/v1.0.0/api/obv.html
                https://github.com/f1tenth/f1tenth_gym/tree/main/examples
"""

import gym
import numpy as np
import yaml
import os

import utils.log as log
from utils.waypoint_loader import WaypointLoader
from controllers.pure_pursuit import PurePursuit, get_lookahead_point
from controllers.lqr_steering import LQRSteeringController
from controllers.lqr_steering_speed import LQRSteeringSpeedController
from utils.rl_utils import get_front_traj, get_interpolated_traj_with_horizon, densify_offset_traj
from utils.render import Renderer, fix_gui


def main():
    method = 'pure_pursuit'  # pure_pursuit, lqr_steering, lqr_steering_speed
    rl_planner = True

    # load map & yaml
    map_name = 'skir'  # levine_2nd, skir, Spielberg, MoscowRaceway, Catalunya
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
    env = gym.make('f110_gym:f110-v0', map=map_path + '/' + map_name + '_map',
                   map_ext='.pgm' if map_name == 'levine_2nd' or map_name == 'skir' else '.png', num_agents=1)
    renderer = Renderer(waypoints)
    env.add_render_callback(renderer.render_waypoints)
    env.add_render_callback(renderer.render_front_traj) if rl_planner else None
    env.add_render_callback(renderer.render_horizon_traj) if rl_planner else None
    env.add_render_callback(renderer.render_lookahead_point) if rl_planner else None
    # env.add_render_callback(fix_gui)
    lap_time = 0.0
    init_pos = np.array([yaml_config['init_pos']])
    obs, _, done, _ = env.reset(init_pos)

    # log_action = []
    # log_obs = []

    horizon = int(10)
    rl_max_speed = 5.0

    while not done:
        if method == 'pure_pursuit' and rl_planner:
            # extract waypoints in coming seconds as front traj
            front_traj = get_front_traj(obs, waypoints, predict_time=1.0)  # [i, x, y, v]
            renderer.front_traj = front_traj
            # print(front_traj.shape)

            # interpolate the front traj to get a 10-point-traj
            horizon_traj = get_interpolated_traj_with_horizon(front_traj, horizon)  # [x, y, v]
            renderer.horizon_traj = horizon_traj

            # TODO: lidar scan & h-traj -> PPO -> lateral offsets
            offset = [0., 0.1, 0.2, 0.3, 0.4, 0.4, 0.3, 0.2, 0.1, 0.0]  # fake offset, [-1, 1], half width [right, left]

            # TODO: interpolate the offsets for every waypoint in traj -> get offset traj in frenet frame
            # TODO: transform it into world frame
            # offset_horizon_traj =   # len = 10

            # interpolate offset traj to get the lookahead point profile (x, y) and v
            dense_offset_traj = densify_offset_traj(horizon_traj)  # [x, y, v]
            lookahead_point_profile = get_lookahead_point(dense_offset_traj, lookahead_dist=1.5)
            renderer.ahead_point = lookahead_point_profile[:2]  # [x, y]
            # print(lookahead_point_profile)

            # TODO: modify PP, input lookahead point, output steering & speed - Derek
            steering, speed = controller.rl_control(obs, lookahead_point_profile, max_speed=rl_max_speed)

        else:
            steering, speed = controller.control(obs)

        print("steering = {}, speed = {}".format(round(steering, 5), round(speed, 5)))
        # log_action.append([lap_time, steering, speed])

        obs, time_step, done, _ = env.step(np.array([[steering, speed]]))
        # log_obs.append([lap_time, obs['poses_x'][0], obs['poses_y'][0],
        #                 obs['poses_theta'][0], obs['linear_vels_x'][0]])

        lap_time += time_step
        env.render(mode='human')

    print('Sim elapsed time:', lap_time)
    # log.xlsx_log_action(method_name, map_name, log_action)
    # log.xlsx_log_observation(method_name, map_name, log_obs)


if __name__ == '__main__':
    main()
