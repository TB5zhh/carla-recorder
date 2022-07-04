# %%
import numpy as np
from PIL import Image
import torch
import re

# %%
V_rgb_path = '/home/ubuntu/tb5zhh/carla/recorder/generated/swingcouch/1/mask_v/1.png'
X_rgb_path = '/home/ubuntu/tb5zhh/carla/recorder/generated/swingcouch/1/mask_x/1.png'
X_depth_path = '/home/ubuntu/tb5zhh/carla/recorder/generated/swingcouch/1/depth_x/1.png'
V_depth_path = '/home/ubuntu/tb5zhh/carla/recorder/generated/swingcouch/1/depth_v/1.png'
find = lambda a: [float(i) for i in re.findall('-?\d+\.\d+', a)]

# pitch yaw roll
# V_pose = torch.tensor([(-41.693489, 39.833096, 2.400506), (0.235068, -90.161369, 0.000013)]).cuda()
# X_pose = torch.tensor([(-41.668877, 23.905540, 4.000000), (-20.000000, 90.000000, 0.000000)]).cuda()
X_pose = torch.tensor(find('Infra Camera: Transform(Location(x=20.235844, y=24.613132, z=4.000000), Rotation(pitch=-15.000000, yaw=0.159198, roll=0.000000))')).reshape((2,-1)).cuda()
V_pose = torch.tensor(find('1: Transform(Location(x=-0.764156, y=24.613132, z=2.794200), Rotation(pitch=0.000000, yaw=0.159198, roll=0.000000))')).reshape((2, -1)).cuda()

# %%
X_depth = torch.as_tensor(np.asarray(Image.open(X_depth_path)), dtype=torch.float64) @ torch.as_tensor(
    (1, 256, 65536), dtype=torch.float64) / (256 * 256 * 256 - 1) * 1000
V_depth = torch.as_tensor(np.asarray(Image.open(V_depth_path)), dtype=torch.float64) @ torch.as_tensor(
    (1, 256, 65536), dtype=torch.float64) / (256 * 256 * 256 - 1) * 1000
V_img = torch.as_tensor(np.asarray(Image.open(V_rgb_path)))
X_img = torch.as_tensor(np.asarray(Image.open(X_rgb_path)))

X_depth = X_depth.cuda()
V_depth = V_depth.cuda()
V_img = V_img.cuda()
X_img = X_img.cuda()


def show_depth(_depth, log=True):
    depth = np.asarray(_depth)
    if log:
        depth = np.log(depth)
    depth = np.clip((depth - depth.min()) / (depth.max() - depth.min()), 0, 1. - 1e-10) * 256.
    depth = np.stack([depth for _ in range(3)], axis=2)
    Image.fromarray(depth.astype(np.uint8)).show()


def vis_3d(_coords, _rgb=None, filename='test.xyz'):
    coords = _coords.reshape((-1, 3))
    if _rgb is not None:
        rgb = _rgb.reshape((-1, 3))
        with open(filename, 'w') as f:
            for p, rgb in zip(coords, rgb):
                print(f"{p[0]} {p[1]} {p[2]} {rgb[0]} {rgb[1]} {rgb[2]}", file=f)
    else:
        with open(filename, 'w') as f:
            for p in coords:
                print(f"{p[0]} {p[1]} {p[2]}", file=f)


show = lambda x: Image.fromarray(np.asarray(x.squeeze())).show()
save = lambda x, path: Image.fromarray(np.asarray(x.squeeze())).save(path)

# %%

# %%
import numpy as np
from PIL import Image
import torch

def convert_depth_to_coords(depth):
    unit_mat = torch.zeros((depth.shape[0], depth.shape[1], 3), device=depth.device)
    unit_mat[:, :, 0] = depth.shape[1] // 2
    unit_mat[:, :, 1] = -(torch.as_tensor(range(0, depth.shape[0])).reshape(
        (-1, 1)).repeat([1, depth.shape[1]]) - depth.shape[0] // 2).to(depth.device)
    unit_mat[:, :, 2] = (torch.as_tensor(range(0, depth.shape[1])).reshape(
        (1, -1)).repeat([depth.shape[0], 1]) - depth.shape[1] // 2).to(depth.device)
    # unit_mat /= np.linalg.norm(unit_mat, axis=2, keepdims=True)
    unit_mat /= depth.shape[1] // 2
    return unit_mat * torch.stack([depth for _ in range(3)], dim=2)


def convert_euler_angles_to_matrix(angles):
    z_axis, y_axis, x_axis = torch.as_tensor(angles[0] / 180 * np.pi,
                                             device=angles.device), torch.as_tensor(angles[1] / 180 * np.pi,
                                                                                    device=angles.device), torch.as_tensor(angles[2] / 180 * np.pi,
                                                                                                                           device=angles.device)
    R_Y = torch.as_tensor([
        [torch.cos(y_axis), 0, torch.sin(y_axis)],
        [0, 1, 0],
        [-torch.sin(y_axis), 0, torch.cos(y_axis)],
    ],
                          dtype=torch.float64,
                          device=angles.device)
    R_Z = torch.as_tensor([
        [torch.cos(z_axis), torch.sin(z_axis), 0],
        [-torch.sin(z_axis), torch.cos(z_axis), 0],
        [0, 0, 1],
    ],
                          dtype=torch.float64,
                          device=angles.device)
    R_X = torch.as_tensor([
        [1, 0, 0],
        [0, torch.cos(x_axis), torch.sin(x_axis)],
        [0, -torch.sin(x_axis), torch.cos(x_axis)],
    ],
                          dtype=torch.float64,
                          device=angles.device)
    return R_Y.T, R_Z.T, R_X.T


# def project(depth, features, src_pose, dst_pose):
depth, features, src_pose, dst_pose = X_depth, X_img, X_pose, V_pose

src_coords = convert_depth_to_coords(depth)
sry, srz, srx = convert_euler_angles_to_matrix(src_pose[1])
dry, drz, drx = convert_euler_angles_to_matrix(dst_pose[1])
src_trans = torch.as_tensor([src_pose[0][0], src_pose[0][2], src_pose[0][1]], device=src_pose.device)
dst_trans = torch.as_tensor([dst_pose[0][0], dst_pose[0][2], dst_pose[0][1]], device=src_pose.device)
coords = (srx @ sry @ srz @ src_coords.reshape((-1, 3)).T).T.reshape(src_coords.shape) + src_trans
dst_coords = (drz.T @ dry.T @ drx.T @ (coords - dst_trans).reshape((-1, 3)).T).T.reshape(coords.shape)
# vis_3d(dst_coords)
dst_coords *= (dst_coords.shape[1] // 2) / torch.abs(dst_coords[:, :, 0:1])
dst_coords[:, :, 1] = dst_coords.shape[0] // 2 - dst_coords[:, :, 1]
dst_coords[:, :, 2] += dst_coords.shape[1] // 2
available = torch.logical_and(
    torch.logical_and(torch.logical_and(
        dst_coords[:, :, 1] < dst_coords.shape[0],
        dst_coords[:, :, 1] >= 0,
    ), torch.logical_and(
        dst_coords[:, :, 2] < dst_coords.shape[1],
        dst_coords[:, :, 2] >= 0,
    )),
    dst_coords[:, :, 0] >= 0,
)
# show(available.cpu())
unique_dst_coords, indices = np.unique(
    np.asarray(dst_coords[available].reshape((-1, 3))[:, 1:].T.cpu()).astype(int),
    axis=1,
    return_index=True,
)

unique_dst_img = torch.as_tensor(features[available].reshape((-1, features.shape[-1])))[indices]
# %%
show(torch.sparse_coo_tensor(unique_dst_coords, available.reshape((-1, 1))[indices], (720, 1280, 1)).to_dense().cpu())
# return torch.sparse_coo_tensor(unique_dst_coords, unique_dst_img, features.shape).to_dense()


# %%
show(project(X_depth, X_img, X_pose, V_pose).cpu())
# %%
for i in range(250):
    V_rgb_path = f'/home/ubuntu/tb5zhh/carla/recorder/generated/swingcouch/1/mask_v/{i+1}.png'
    X_rgb_path = f'/home/ubuntu/tb5zhh/carla/recorder/generated/swingcouch/1/mask_x/{i+1}.png'
    X_depth_path = f'/home/ubuntu/tb5zhh/carla/recorder/generated/swingcouch/1/depth_x/{i+1}.png'
    V_depth_path = f'/home/ubuntu/tb5zhh/carla/recorder/generated/swingcouch/1/depth_v/{i+1}.png'
    find = lambda a: [float(i) for i in re.findall('-?\d+\.\d+', a)]
    with open('/home/ubuntu/tb5zhh/carla/recorder/generated/swingcouch/1/path.txt') as f:
        lines = f.readlines()
    X_pose = torch.tensor(find(lines[0])).reshape((2,-1)).cuda()
    V_pose = torch.tensor(find(lines[i+1])).reshape((2, -1)).cuda()

    X_depth = torch.as_tensor(np.asarray(Image.open(X_depth_path)), dtype=torch.float64) @ torch.as_tensor(
        (1, 256, 65536), dtype=torch.float64) / (256 * 256 * 256 - 1) * 1000
    V_depth = torch.as_tensor(np.asarray(Image.open(V_depth_path)), dtype=torch.float64) @ torch.as_tensor(
        (1, 256, 65536), dtype=torch.float64) / (256 * 256 * 256 - 1) * 1000
    V_img = torch.as_tensor(np.asarray(Image.open(V_rgb_path)))
    X_img = torch.as_tensor(np.asarray(Image.open(X_rgb_path)))

    X_depth = X_depth.cuda()
    V_depth = V_depth.cuda()
    V_img = V_img.cuda()
    X_img = X_img.cuda()
    save(project(X_depth, X_img, X_pose, V_pose).cpu(), f'tmp/{i+1}.png')
# %%
