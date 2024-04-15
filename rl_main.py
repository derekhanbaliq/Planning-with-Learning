"""
    main application for RL planner
    Author: Derek Zhou, Biao Wang, Tian Tan
    References: https://f1tenth-gym.readthedocs.io/en/v1.0.0/api/obv.html
                https://github.com/f1tenth/f1tenth_gym/tree/main/examples
"""

import os
import numpy as np
import yaml

from controllers.pure_pursuit import PurePursuit, get_lookahead_point
from utils.render import Renderer, fix_gui
from utils.rl_utils import get_front_traj, get_interpolated_traj_with_horizon, densify_offset_traj
from utils.rl_utils import get_offset_traj
from utils.waypoint_loader import WaypointLoader

from f110_env_rl import F110RLEnv


def main():
    # pure_pursuit + enable_rl_planner

    # load map & yaml
    # map_name = 'levine_2nd'  # levine_2nd, skir, Spielberg, MoscowRaceway, Catalunya
    # map_path = os.path.abspath(os.path.join('maps', map_name))
    # yaml_config = yaml.load(open(map_path + '/' + map_name + '_map.yaml'), Loader=yaml.FullLoader)

    # # load waypoints
    # csv_data = np.loadtxt(map_path + '/' + map_name + '_raceline.csv', delimiter=';', skiprows=0)  # '_centerline.csv'
    # waypoints = WaypointLoader(map_name, csv_data)

    # # load controller
    # controller = PurePursuit(waypoints)

    # env = F110RLEnv(seed=0, map=map_path + '/' + map_name + '_map',
    #                 map_ext='.pgm' if map_name == 'levine_2nd' or map_name == 'skir' else '.png', num_agents=1)

    env = F110RLEnv()

    # renderer = Renderer(waypoints)
    # env.add_render_callback(renderer.render_waypoints)
    # env.add_render_callback(renderer.render_front_traj)
    # env.add_render_callback(renderer.render_horizon_traj)
    # env.add_render_callback(renderer.render_lookahead_point)
    # env.add_render_callback(fix_gui)
    # lap_time = 0.0
    # init_pos = np.array([yaml_config['init_pos']])
    # obs, _, done, _ = env.reset(init_pos)

    # obs = env.obs
    # horizon = int(10)
    # rl_max_speed = 5.0

    # done = False

    while not env.done:
        env.env_step()
        # # extract waypoints in coming seconds as front traj
        # front_traj = get_front_traj(env.obs, env.waypoints, predict_time=1.0)  # [i, x, y, v]
        # env.renderer.front_traj = front_traj
        # # print(front_traj.shape)

        # # interpolate the front traj to get a 10-point-traj
        # horizon_traj = get_interpolated_traj_with_horizon(front_traj, horizon)  # [x, y, v]
        # env.renderer.horizon_traj = horizon_traj

        # # TODO: lidar scan & h-traj -> PPO -> lateral offsets
        # # offset = PPO_model(obs['scan'], horizon_traj)
        # offset = [0., 0.1, 0.2, 0.3, 0.4, 0.4, 0.3, 0.2, 0.1, 0.0]  # fake offset, [-1, 1], half width [right, left]

        # # TODO: interpolate the offsets for every waypoint in traj -> get offset traj in frenet frame
        # # TODO: transform it into world frame
        # # offset_horizon_traj =   # len = 10

        # # interpolate offset traj to get the lookahead point profile (x, y) and v
        # dense_offset_traj = densify_offset_traj(horizon_traj)  # [x, y, v]
        # lookahead_point_profile = get_lookahead_point(dense_offset_traj, lookahead_dist=1.5)
        # env.renderer.ahead_point = lookahead_point_profile[:2]  # [x, y]

        # # input lookahead point pos & ref speed, output steering & speed
        # steering, speed = env.controller.rl_control(env.obs, lookahead_point_profile, max_speed=rl_max_speed)

        # env.obs, time_step, env.done, _ = env.env_step(np.array([[steering, speed]]))

        # lap_time += time_step

    print('Sim elapsed time:', env.lap_time)


if __name__ == '__main__':
    main()
