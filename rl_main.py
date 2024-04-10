"""
    main application for RL planner
    Author: Derek Zhou
    References: https://f1tenth-gym.readthedocs.io/en/v1.0.0/api/obv.html
                https://github.com/f1tenth/f1tenth_gym/tree/main/examples
"""

import gym
import numpy as np
import yaml
import os

import utils.log as log
from utils.waypoint_loader import WaypointLoader
from controllers.pure_pursuit import PurePursuit
from controllers.lqr_steering import LQRSteeringController
from controllers.lqr_steering_speed import LQRSteeringSpeedController
from utils.rl_utils import get_front_traj, get_interpolated_traj_with_horizon
from utils.render import Renderer


def main():
    method = 'pure_pursuit'  # pure_pursuit, lqr_steering, lqr_steering_speed

    # load map & yaml
    map_name = 'MoscowRaceway'  # Spielberg, example, MoscowRaceway, Catalunya, levine
    map_path = os.path.abspath(os.path.join('maps', map_name))
    yaml_config = yaml.load(open(map_path + '/' + map_name + '_map.yaml'), Loader=yaml.FullLoader)

    # load waypoints
    csv_data = np.loadtxt(map_path + '/' + map_name + '_raceline.csv', delimiter=';', skiprows=0)
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
    env = gym.make('f110_gym:f110-v0', map=map_path + '/' + map_name + '_map', map_ext='.png', num_agents=1)  # .pgm
    renderer = Renderer(waypoints)
    env.add_render_callback(renderer.render_waypoints)
    env.add_render_callback(renderer.render_traj) # render the reference trajectory
    lap_time = 0.0
    init_pos = np.array([yaml_config['init_pos']])
    obs, _, done, _ = env.reset(init_pos)

    # log_action = []
    # log_obs = []

    horizon = 10

    while not done:
        front_traj = get_front_traj(obs, waypoints, predict_time=2)  # [i, x, y, v]
        print(front_traj.shape)

        # TODO: visualize traj in gui - Biao

        # TODO: interpolate traj to a fixed num - horizon - Derek
        horizon_traj = get_interpolated_traj_with_horizon(front_traj, horizon)  # [x, y, v]
        print(horizon_traj.shape)
        
        # update the reference trajectory for rendering
        renderer.traj = front_traj

        # TODO: lidar scan & h-traj -> PPO -> lateral offsets !!!!
        offset = [0., 0.1, 0.2, 0.3, 0.4, 0.4, 0.3, 0.2, 0.1, 0.0]  # fake offset, [-1, 1], half width [right, left]

        # TODO: interpolate the offsets for every waypoint in traj -> get offsetted traj in frenet frame
        # TODO: transform it into world frame
        # offset_horizon_traj =   # len = 10

        # TODO: extract lookahead point
        # interpolate the lookahead point + corresponding speed into the offset_horizon_traj curve

        # TODO: modify PP, input lookahead point, output steering & speed - Derek

        steering, speed = controller.control(obs)  # each agent’s current observation
        print("steering = {}, speed = {}".format(round(steering, 5), speed))
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
