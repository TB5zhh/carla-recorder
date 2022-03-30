# %%
V_rgb_path = '/home/ubuntu/tb5zhh/carla/recorder/ADtiao/12/rgb_v/100.png'
X_rgb_path = '/home/ubuntu/tb5zhh/carla/recorder/ADtiao/12/rgb_x/100.png'
X_depth_path = '/home/ubuntu/tb5zhh/carla/recorder/ADtiao/12/depth_x/100.png'

# pitch yaw roll
V_pose = [(-45.109760, 71.010406, 2.400504), (0.097392, 90.088753, 0.001260)]
X_pose = [(-45., 90., 5.), (0., -90., 0.)]

import numpy as np
from PIL import Image

# %%
X_depth = np.asarray(Image.open(X_depth_path), dtype=np.float64) @ (1, 256, 65536) / 1000
# X_depth = np.log(X_depth)
# X_depth = np.clip((X_depth - X_depth.min()) / (X_depth.max() - X_depth.min()), 0, 1.- 1e-10) * 256.
# X_depth = np.stack([X_depth for _ in range(3)], axis=2)
# Image.fromarray(X_depth.astype(np.uint8)).save('test.png')
# %%

unit_mat = np.zeros((X_depth.shape[0], X_depth.shape[1], 3))

unit_mat[:, :, 0] = np.repeat(np.array(range(X_depth.shape[0])).reshape((-1, 1)), X_depth.shape[1], axis=1)
unit_mat[:, :, 1] = np.repeat(np.array(range(X_depth.shape[1])).reshape((1, -1)), X_depth.shape[0], axis=0)
unit_mat[:, :, 2] = X_depth.shape[1] // 2
unit_mat /= np.linalg.norm(unit_mat, axis=2, keepdims=True)
X_depth * unit_mat + X_pose[0]

# %%
