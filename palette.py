PALETTE = [
    [  0,   0,   0], # unlabeled     =   0,
    [ 70,  70,  70], # building      =   1,
    [100,  40,  40], # fence         =   2,
    [ 55,  90,  80], # other         =   3,
    [220,  20,  60], # pedestrian    =   4,
    [153, 153, 153], # pole          =   5,
    [157, 234,  50], # road line     =   6,
    [128,  64, 128], # road          =   7,
    [244,  35, 232], # sidewalk      =   8,
    [107, 142,  35], # vegetation    =   9,
    [  0,   0, 142], # vehicle       =  10,
    [102, 102, 156], # wall          =  11,
    [220, 220,   0], # traffic sign  =  12,
    [ 70, 130, 180], # sky           =  13,
    [ 81,   0,  81], # ground        =  14,
    [150, 100, 100], # bridge        =  15,
    [230, 150, 140], # rail track    =  16,
    [180, 165, 180], # guard rail    =  17,
    [250, 170,  30], # traffic light =  18,
    [110, 190, 160], # static        =  19,
    [170, 120,  50], # dynamic       =  20,
    [ 45,  60, 150], # water         =  21,
    [145, 170, 100], # terrain       =  22,
    [236, 236, 236], # general anomaly = 23,
]

import numpy as np
# import torch
from PIL import Image

def convert_rgb_to_mask(rgb):
    """Convert H * W * 3 color map to H * W mask

    Args:
        rgb (torch.Tensor): Color map, torch.Tensor with shape of H * W * 3

    Returns:
        torch.Tensor: Semantic mask, torch.Tensor with shape of H * W
    """
    assert len(rgb.shape) == 3 and rgb.shape[2] == 3, "Input tensor must be a tensor of H * W * 3"
    h, w = rgb.shape[:2]
    # Image.fromarray(np.asarray(rgb)).save('test.png')
    im = rgb.reshape((-1, 3))
    result = 255 * np.ones(im.shape[0], dtype=np.uint8)
    for cateid, catergb in enumerate(PALETTE):
        val = np.array(catergb, dtype=np.uint8)
        mask = (im == val).all(axis=1)
        result[mask] = cateid
    result = result.reshape((h, w))
    # Image.fromarray(np.asarray(result)).save('test1.png')
    return result

def convert_mask_to_rgb(mask):
    """Convert H * W mask to H * W * 3 color map

    Args:
        mask (torch.Tensor): Semantic mask, torch.Tensor with shape of H * W

    Returns:
        torch.Tensor: Color map, torch.Tensor with shape of H * W * 3
    """
    assert len(mask.shape) == 2, 'Input tensor must be a tensor of H * W'
    h, w = mask.shape
    # Image.fromarray(np.asarray(mask)).save('test.png')
    im = mask.flatten()
    result = np.zeros((im.shape[0], 3), dtype=np.uint8)
    result[:, 0] = 255
    for cateid, catergb in enumerate(PALETTE):
        val = np.array(catergb, dtype=np.uint8)
        mask = (im == cateid)
        result[mask, :] = val
    result = result.reshape((h, w, 3))
    # Image.fromarray(np.asarray(result)).save('test2.png')
    return result