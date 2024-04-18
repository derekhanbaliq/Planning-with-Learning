import numpy as np
import matplotlib.pyplot as plt


def add_lateral_offset2get_new_traj(traj, offset):
    traj = np.array(traj)
    offset = np.asarray(offset)

    xy_traj = traj[:, :2]

    tangents = np.zeros_like(xy_traj)
    tangents[0] = xy_traj[1] - xy_traj[0]
    tangents[-1] = xy_traj[-1] - xy_traj[-2]
    tangents[1:-1] = xy_traj[2:] - xy_traj[:-2]

    norms = np.linalg.norm(tangents, axis=1)
    unit_tangents = tangents / norms[:, np.newaxis]

    normals = np.array([-unit_tangents[:, 1], unit_tangents[:, 0]]).T
    normal_norms = np.linalg.norm(normals, axis=1)
    unit_normals = normals / normal_norms[:, np.newaxis]

    offsets_rescaled = offset[:, np.newaxis] * unit_normals
    new_xy_traj = xy_traj + offsets_rescaled

    new_traj = np.hstack([new_xy_traj, traj[:, 2:]])
    return new_traj, unit_normals, offsets_rescaled


traj = np.array([
    [9.263957, 7.4023188, 4.4782284],
    [9.00371679, 7.86495135, 4.24964586],
    [8.63561439, 8.24645418, 4.23776537],
    [8.179714, 8.5180591, 4.4067167],
    [7.67687793, 8.68927809, 4.71947207],
    [7.15318939, 8.7803028, 5.1295941],
    [6.6226076, 8.8159781, 5.6531315],
    [6.09075985, 8.82356962, 6.33136446],
    [5.5588414, 8.82455123, 7.08846069],
    [5.026922, 8.8248816, 7.789943]
])
offset = np.array([0.5, 0.5, 0.2, 0.3, 0.4, 0.4, 0.3, 0.2, 0.1, 0.1])

new_traj, unit_normals, offsets_rescaled = add_lateral_offset2get_new_traj(traj, offset)

fig, ax = plt.subplots(figsize=(8, 3))
ax.plot(traj[:, 0], traj[:, 1], 'o-', label='Original Trajectory')
ax.plot(new_traj[:, 0], new_traj[:, 1], 's-', label='Trajectory with Lateral Offset')
for point, magnitude in zip(traj, offsets_rescaled):
    ax.arrow(point[0], point[1], magnitude[0], magnitude[1], head_width=0.05, head_length=0.1, fc='red', ec='red')

ax.set_xlabel('X Coordinate')
ax.set_ylabel('Y Coordinate')
ax.legend()
plt.grid(True)
plt.show()
