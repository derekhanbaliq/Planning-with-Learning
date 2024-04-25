"""
    inference for testing the algorithms
    Author: Derek Zhou, Biao Wang, Tian Tan
    References: https://f1tenth-gym.readthedocs.io/en/v1.0.0/api/obv.html
                https://github.com/f1tenth/f1tenth_gym/tree/main/examples
"""

import os

import numpy as np
import torch

from controllers.lqr_steering import LQRSteeringController
from controllers.lqr_steering_speed import LQRSteeringSpeedController
from controllers.pure_pursuit import PurePursuit, get_lookahead_point
from f110_env_rl import F110RLEnv
from f110_gym.envs.f110_env import F110Env
from ppo_continuous import Agent
from utils.render import Renderer, fix_gui
from utils.traj_utils import get_front_traj, get_interpolated_traj_with_horizon, densify_offset_traj, get_offset_traj, \
                             global_to_local, local_to_global, bresenham_line_index, bresenham_line_point
from utils.waypoint_loader import WaypointLoader
from utils.lidar_utils import downsample_lidar_scan, get_lidar_data


from f110_gym.envs.base_classes import RaceCar


def main():
    method = 'pure_pursuit'  # pure_pursuit, lqr_steering, lqr_steering_speed
    rl_planner = True  # if use RL planner, then enable

    # load map & waypoints
    map_name = 'skir'  # levine_2nd, skir, Spielberg, MoscowRaceway, Catalunya
    map_name = 'skir'  # levine_2nd, skir, Spielberg, MoscowRaceway, Catalunya
    map_path = os.path.abspath(os.path.join('maps', map_name))
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

    # generate random obstacles
    num_obstacles = 0  # length_obs = 0.58 # 0.32, width_obs  = 0.31 # 0.22
    obt_index = np.random.uniform(1, waypoints.x.shape[0]-1, size=(num_obstacles,)).astype(int)
    
    # obt_pose = np.array([waypoints.x[obt_index], waypoints.y[obt_index], waypoints.θ[obt_index]]).transpose().reshape((-1, 3))
    thetas = np.arctan2(waypoints.x[obt_index+1]-waypoints.x[obt_index-1], waypoints.y[obt_index+1]-waypoints.y[obt_index-1])
    obt_pose = np.array([waypoints.x[obt_index], waypoints.y[obt_index], thetas]).transpose().reshape((-1, 3))

    # create & init env
    env = F110Env(map=map_path + '/' + map_name + '_map',
                  map_ext='.pgm' if map_name == 'levine_2nd' or map_name == 'skir' else '.png', num_agents=1,
                  obt_poses=obt_pose)

    renderer = Renderer(waypoints)
    env.add_render_callback(renderer.render_waypoints)
    # env.add_render_callback(renderer.render_front_traj) if rl_planner else None
    # env.add_render_callback(renderer.render_horizon_traj) if rl_planner else None
    env.add_render_callback(renderer.render_lookahead_point) if rl_planner else None
    env.add_render_callback(renderer.render_offset_traj) if rl_planner else None
    env.add_render_callback(renderer.render_lidar_data) if rl_planner else None
    env.add_render_callback(fix_gui)

    lap_time = 0.0

    init_index = np.random.randint(0, waypoints.x.shape[0])
    init_pos = np.array([waypoints.x[init_index], waypoints.y[init_index], waypoints.θ[init_index]]).reshape((1, -1))
    # print("init index = {}, init pose = {}".format(init_index, init_pos))

    obs, _, done, _ = env.reset(init_pos)

    rl_env = F110RLEnv(render=False, map_name=map_name, num_obstacles=num_obstacles, obt_poses=obt_pose,
                       num_lidar_scan=1080)
    model = Agent(rl_env)
    model.load_state_dict(torch.load('skir_with_obs_0.pkl'))

    while not done:
        if method == 'pure_pursuit' and rl_planner:
            # lidar data for further usage
            downsampled_lidar_scan = downsample_lidar_scan(obs['scans'][0].flatten(), rl_env.num_lidar_scan).flatten()
            lidar_data = get_lidar_data(downsampled_lidar_scan, obs['poses_x'], obs['poses_y'], obs['poses_theta'])

            # extract waypoints in predicted time & interpolate the front traj to get a 10-point-traj
            front_traj = get_front_traj(obs, waypoints, predict_time=rl_env.predict_time)  # [i, x, y, v]
            horizon_traj = get_interpolated_traj_with_horizon(front_traj, rl_env.horizon)  # [x, y, v]
            local_horizon_traj = global_to_local(obs, horizon_traj)

            # gather agent obs & infer the action
            network_obs = np.hstack((downsampled_lidar_scan,
                                     local_horizon_traj[:, :2].flatten(),
                                     np.array([obs['poses_x'][0], obs['poses_y'][0]]).reshape((-1,))))
            network_obs = torch.from_numpy(network_obs).to(dtype=torch.float).resize(1, network_obs.shape[0])
            action, sum_log_prob, sum_entropy, value = model.get_action_and_value(network_obs)  # PPO inference
            offset = action.numpy().flatten()
            print(offset)

            # add offsets on horizon traj & densify offset traj to 80 points & get lookahead point & pure pursuit
            local_offset_traj = get_offset_traj(local_horizon_traj, offset)
            offset_traj = local_to_global(obs, local_offset_traj)
            dense_offset_traj = densify_offset_traj(offset_traj)  # [x, y, v]
            lookahead_point_profile = get_lookahead_point(dense_offset_traj, lookahead_dist=1.5)
            steering, speed = controller.rl_control(obs, lookahead_point_profile, max_speed=rl_env.rl_max_speed)
            
            
            # offset_x_index = np.ceil((offset_traj[:,0] + 12) / 0.05).astype(int)
            # offset_y_index = np.ceil((offset_traj[:,1] + 10.7) / 0.05).astype(int)
            # offset_traj_indices = np.vstack((offset_x_index, offset_y_index)).T
            
            # max_rows = RaceCar.scan_simulator.map_img.shape[0]
            # max_cols = RaceCar.scan_simulator.map_img.shape[1]
            
            # all_indices = []
            # for i in range(offset_traj_indices.shape[0]-1):
            #     line_indices = bresenham_line_index(offset_traj_indices[i, :], offset_traj_indices[i+1, :])
            #     all_indices.append(line_indices)
            # all_indices = np.concatenate(all_indices).reshape(-1, 2)
            
            # filtered_traj_indices = all_indices[(all_indices[:,1]<max_rows) & (all_indices[:,0]<max_cols)]
            # print("offset traj index:", filtered_traj_indices.shape)
            # # print(RaceCar.scan_simulator.map_img.shape)
            # print(np.count_nonzero(RaceCar.scan_simulator.map_img[filtered_traj_indices[:,1], filtered_traj_indices[:,0]]==0))
            
            # all_points = []
            # for i in range(offset_traj.shape[0]-1):
            #     line_points = bresenham_line_point(offset_traj[i, :], offset_traj[i+1, :])
            #     all_points.append(line_points)
            # all_points = np.concatenate(all_points).reshape(-1, 2)
            # print("line points: ", all_points.shape)

            renderer.lidar_data = lidar_data
            # renderer.front_traj = front_traj
            # renderer.horizon_traj = horizon_traj
            # renderer.horizon_traj = all_points
            renderer.offset_traj = offset_traj
            renderer.ahead_point = lookahead_point_profile[:2]  # [x, y]

        else:
            steering, speed = controller.control(obs)

        print("steering = {}, speed = {}".format(round(steering, 4), round(speed, 4)))

        obs, time_step, done, _ = env.step(np.array([[steering, speed]]))

        lap_time += time_step
        env.render(mode='human')

    print('Sim elapsed time:', lap_time)


if __name__ == '__main__':
    main()
