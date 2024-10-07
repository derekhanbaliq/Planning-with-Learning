"""
    inference for testing the algorithms
    Author: Derek Zhou, Biao Wang
    References: https://f1tenth-gym.readthedocs.io/en/v1.0.0/api/obv.html
                https://github.com/f1tenth/f1tenth_gym/tree/main/examples
"""

import os

import numpy as np
import torch

from controllers.lqr_steering import LQRSteeringController
from controllers.lqr_steering_speed import LQRSteeringSpeedController
from controllers.pure_pursuit import PurePursuit, get_lookahead_point
from controllers.kmpc import MPCConfig_F110, KinematicModel, KMPCController
from f110_env_rl import F110RLEnv
from f110_gym.envs.base_classes import RaceCar
from f110_gym.envs.f110_env import F110Env
from ppo_continuous import Agent
from utils.lidar_utils import downsample_lidar_scan, get_lidar_data
from utils.render import Renderer, fix_gui
from utils.traj_utils import get_front_traj, get_interpolated_traj_with_horizon, densify_offset_traj, get_offset_traj, \
    global_to_local, local_to_global, bresenham_line_index, bresenham_line_point
from utils.waypoint_loader import WaypointLoader, pi_2_pi, waypoints_dir_correction


def main():
    method = 'pure_pursuit'  # !!!! pure_pursuit, lqr_steering, lqr_steering_speed, kinematic_mpc
    rl_planner = True  # !!!! enable if you use RL planner

    # load map & waypoints
    map_name = 'skir_blocked'  # !!!! levine_2nd, skir, skir_blocked, Spielberg, MoscowRaceway, Catalunya
    map_path = os.path.abspath(os.path.join('maps', map_name))
    csv_data = np.loadtxt(map_path + '/' + map_name + '_raceline.csv', delimiter=';', skiprows=0)  # '_centerline.csv'
    waypoints = WaypointLoader(map_name, csv_data)

    # load controller
    controller = None
    if method == 'pure_pursuit':
        controller = PurePursuit(waypoints, L=0.8)  # !!!!
    elif method == 'lqr_steering':
        controller = LQRSteeringController(waypoints)
    elif method == 'lqr_steering_speed':
        controller = LQRSteeringSpeedController(waypoints)
    elif method == 'kinematic_mpc':
        model_config = MPCConfig_F110()
        kmpc_waypoints = waypoints_dir_correction(map_name, csv_data)
        controller = KMPCController(model=KinematicModel(config=model_config), waypoints=kmpc_waypoints,
                                    config=model_config)
        control_period = 10  # Hz

    # generate random obstacles
    num_obstacles = 0  # length_obs = 0.58 # 0.32, width_obs  = 0.31 # 0.22
    obt_index = np.random.uniform(1, waypoints.x.shape[0] - 1, size=(num_obstacles,)).astype(int)
    thetas = np.arctan2(waypoints.x[obt_index + 1] - waypoints.x[obt_index - 1],
                        waypoints.y[obt_index + 1] - waypoints.y[obt_index - 1])
    obt_pose = np.array([waypoints.x[obt_index], waypoints.y[obt_index], thetas]).transpose().reshape((-1, 3))

    # create & init env
    env = F110Env(map=map_path + '/' + map_name + '_map',
                  map_ext='.pgm' if map_name == 'levine_2nd' or map_name == 'skir' or map_name == 'skir_blocked' else '.png',
                  num_agents=1, obt_poses=obt_pose)

    renderer = Renderer(waypoints)
    env.add_render_callback(renderer.render_waypoints)
    # env.add_render_callback(renderer.render_front_traj) if rl_planner else None
    env.add_render_callback(renderer.render_horizon_traj) if rl_planner or method == 'kinematic_mpc' else None
    env.add_render_callback(renderer.render_lookahead_point) if rl_planner and method == 'pure_pursuit' else None
    env.add_render_callback(renderer.render_offset_traj) if rl_planner or method == 'kinematic_mpc' else None
    env.add_render_callback(renderer.render_lidar_data) if rl_planner else None
    # env.add_render_callback(fix_gui)

    lap_time = 0.0
    init_index = np.random.randint(0, waypoints.x.shape[0])
    init_pos = np.array([waypoints.x[init_index], waypoints.y[init_index], waypoints.θ[init_index]]).reshape((1, -1))
    # print("init index = {}, init pose = {}".format(init_index, init_pos))
    # init_pos = np.array([[0.0, 0.0, 0.0]])  # fixed init or not

    obs, _, done, _ = env.reset(init_pos)

    rl_env = F110RLEnv(render=False, map_name=map_name, num_obstacles=num_obstacles, obt_poses=obt_pose,
                       num_lidar_scan=108, ctrl_method=method)
    model = Agent(rl_env)
    model.load_state_dict(torch.load(f'pkls/nudge_2s_4obs_10m.pkl'))  # !!!! modify load model

    while not done:
        if method == 'pure_pursuit' and rl_planner:
            # lidar data for further usage
            downsampled_lidar_scan = downsample_lidar_scan(obs['scans'][0].flatten(), rl_env.num_lidar_scan).flatten()
            lidar_data = get_lidar_data(downsampled_lidar_scan, obs['poses_x'], obs['poses_y'], obs['poses_theta'])

            # extract waypoints in predicted time & interpolate the front traj to get a 10-point-traj
            front_traj = get_front_traj(obs, waypoints, predict_time=rl_env.predict_time)  # [i, x, y, v, θ]
            horizon_traj = get_interpolated_traj_with_horizon(front_traj, rl_env.horizon)  # [x, y, v, θ]
            local_horizon_traj = global_to_local(obs, horizon_traj)  # [x, y, v, θ]

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
            offset_traj = np.vstack((np.array([[obs['poses_x'][0], obs['poses_y'][0], 2.0, obs['poses_theta'][0]]]),
                                     offset_traj))  # add car pose as the first point
            dense_offset_traj = densify_offset_traj(offset_traj)  # [x, y, v, theta]
            lookahead_point_profile = get_lookahead_point(obs, dense_offset_traj, lookahead_dist=rl_env.lookahead_dist)
            steering, speed = controller.rl_control(obs, lookahead_point_profile, max_speed=rl_env.fixed_speed)

            offset_x_index = np.ceil((offset_traj[:, 0] + 12) / 0.05).astype(int)
            offset_y_index = np.ceil((offset_traj[:, 1] + 10.7) / 0.05).astype(int)
            offset_traj_indices = np.vstack((offset_x_index, offset_y_index)).T
            max_rows = RaceCar.scan_simulator.map_img.shape[0]
            max_cols = RaceCar.scan_simulator.map_img.shape[1]
            all_indices = []
            for i in range(offset_traj_indices.shape[0] - 1):
                line_indices = bresenham_line_index(offset_traj_indices[i, :], offset_traj_indices[i + 1, :])
                all_indices.append(line_indices)
            all_indices = np.concatenate(all_indices).reshape(-1, 2)
            filtered_traj_indices = all_indices[(all_indices[:, 1] < max_rows) & (all_indices[:, 0] < max_cols) &
                                                (all_indices[:, 1] >= 0) & (all_indices[:, 0] >= 0)]
            # print("offset traj index:", filtered_traj_indices.shape)
            # print(RaceCar.scan_simulator.map_img.shape)
            # print("number of overlapped points =", np.count_nonzero(
            #     RaceCar.scan_simulator.map_img[filtered_traj_indices[:, 1], filtered_traj_indices[:, 0]] == 0))
            all_points = []
            for i in range(offset_traj.shape[0] - 1):
                line_points = bresenham_line_point(offset_traj[i, :], offset_traj[i + 1, :])
                all_points.append(line_points)
            all_points = np.concatenate(all_points).reshape(-1, 2)
            # print("line points: ", all_points.shape)

            renderer.lidar_data = lidar_data
            # renderer.front_traj = front_traj
            # renderer.horizon_traj = horizon_traj
            renderer.horizon_traj = all_points
            renderer.offset_traj = offset_traj
            renderer.ahead_point = lookahead_point_profile[:2]  # [x, y]

        elif method == 'kinematic_mpc' and not rl_planner:
            if int(lap_time * 100) % control_period == 0:  # 50 ms
                veh_state = np.array([env.sim.agents[0].state[0],
                                      env.sim.agents[0].state[1],
                                      env.sim.agents[0].state[3],  # vx
                                      env.sim.agents[0].state[4],  # yaw angle
                                      ])
                steering, speed, ref_path_x, ref_path_y, pred_x, pred_y, mpc_ox, mpc_oy, a = controller.control(veh_state)
                renderer.offset_traj = np.array([ref_path_x, ref_path_y]).T  # red
                renderer.horizon_traj = np.array([pred_x, pred_y]).T  # yellow

        elif method == 'kinematic_mpc' and rl_planner:
            # lidar data for further usage
            downsampled_lidar_scan = downsample_lidar_scan(obs['scans'][0].flatten(), rl_env.num_lidar_scan).flatten()
            lidar_data = get_lidar_data(downsampled_lidar_scan, obs['poses_x'], obs['poses_y'], obs['poses_theta'])

            # extract waypoints in predicted time & interpolate the front traj to get a 10-point-traj
            front_traj = get_front_traj(obs, waypoints, predict_time=1.0)  # [i, x, y, v, θ], 10Hz * 10 horizon
            horizon_traj = get_interpolated_traj_with_horizon(front_traj, rl_env.horizon)  # [x, y, v, θ]
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
            mpc_offset_traj = densify_offset_traj(offset_traj, intep_num=11)

            if int(lap_time * 100) % control_period == 0:  # 50 ms
                veh_state = np.array([env.sim.agents[0].state[0],
                                      env.sim.agents[0].state[1],
                                      env.sim.agents[0].state[3],  # vx
                                      env.sim.agents[0].state[4],  # yaw angle
                                      ])
                steering, speed, ref_path_x, ref_path_y, pred_x, pred_y, mpc_ox, mpc_oy, a = controller.rl_control(veh_state, mpc_offset_traj)
                renderer.offset_traj = np.array([ref_path_x, ref_path_y]).T  # red
                renderer.horizon_traj = np.array([pred_x, pred_y]).T  # yellow

            # offset_x_index = np.ceil((offset_traj[:, 0] + 12) / 0.05).astype(int)
            # offset_y_index = np.ceil((offset_traj[:, 1] + 10.7) / 0.05).astype(int)
            # offset_traj_indices = np.vstack((offset_x_index, offset_y_index)).T
            # max_rows = RaceCar.scan_simulator.map_img.shape[0]
            # max_cols = RaceCar.scan_simulator.map_img.shape[1]
            # all_indices = []
            # for i in range(offset_traj_indices.shape[0] - 1):
            #     line_indices = bresenham_line_index(offset_traj_indices[i, :], offset_traj_indices[i + 1, :])
            #     all_indices.append(line_indices)
            # all_indices = np.concatenate(all_indices).reshape(-1, 2)
            # filtered_traj_indices = all_indices[(all_indices[:, 1] < max_rows) & (all_indices[:, 0] < max_cols) &
            #                                     (all_indices[:, 1] >= 0) & (all_indices[:, 0] >= 0)]
            # # print("offset traj index:", filtered_traj_indices.shape)
            # # print(RaceCar.scan_simulator.map_img.shape)
            # print("number of overlapped points =", np.count_nonzero(
            #     RaceCar.scan_simulator.map_img[filtered_traj_indices[:, 1], filtered_traj_indices[:, 0]] == 0))
            # all_points = []
            # for i in range(offset_traj.shape[0] - 1):
            #     line_points = bresenham_line_point(offset_traj[i, :], offset_traj[i + 1, :])
            #     all_points.append(line_points)
            # all_points = np.concatenate(all_points).reshape(-1, 2)
            # print("line points: ", all_points.shape)

            renderer.lidar_data = lidar_data

        else:
            steering, speed = controller.control(obs)

        print("steering = {}, speed = {}".format(round(steering, 4), round(speed, 4)))
        obs, time_step, done, _ = env.step(np.array([[steering, speed]]))
        lap_time += time_step
        env.render(mode='human')

    print('Sim elapsed time:', lap_time)


if __name__ == '__main__':
    main()
