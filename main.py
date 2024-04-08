"""
    main application of controllers
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
from utils.render import Renderer


def main():
    method = 'lqr_steering_speed'  # pure_pursuit, lqr_steering, lqr_steering_speed

    # load map & yaml
    map_name = 'MoscowRaceway'  # Spielberg, example, MoscowRaceway, Catalunya, levine
    map_path = os.path.abspath(os.path.join('maps', map_name))
    yaml_config = yaml.load(open(map_path + '/' + map_name + '_map.yaml'), Loader=yaml.FullLoader)

    # load waypoints
    csv_data = np.loadtxt(map_path + '/' + map_name + '_raceline.csv', delimiter=';', skiprows=0)
    waypoints = WaypointLoader(map_name, csv_data)

    # load controller
    if method == 'pure_pursuit':
        controller = PurePursuit(waypoints)
    elif method == 'lqr_steering':
        controller = LQRSteeringController(waypoints)
    elif method == 'lqr_steering_speed':
        controller = LQRSteeringSpeedController(waypoints)

    # create env & init
    env = gym.make('f110_gym:f110-v0', map=map_path + '/' + map_name + '_map', map_ext='.png', num_agents=1)
    # env = gym.make('f110_gym:f110-v0', map=map_path + '/' + map_name + '_map', map_ext='.pgm', num_agents=1)  # levine
    renderer = Renderer(waypoints)
    env.add_render_callback(renderer.render_waypoints)

    # init
    init_pos = np.array([yaml_config['init_pos']])
    obs, _, done, _ = env.reset(init_pos)
    lap_time = 0.0

    # log_action = []
    # log_obs = []

    while not done:
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
