PALETTE = [
    [[128,  64, 128], [157, 234,  50]], # road          =   7,
    [244,  35, 232], # sidewalk      =   8,
    [ 70,  70,  70], # building      =   1,
    [102, 102, 156], # wall          =  11,
    [100,  40,  40], # fence         =   2,
    [153, 153, 153], # pole          =   5,
    [250, 170,  30], # traffic light =  18,
    [220, 220,   0], # traffic sign  =  12,
    [107, 142,  35], # vegetation    =   9,
    [[145, 170, 100],[150, 100, 100]], # terrain       =  22,
    [ 70, 130, 180], # sky           =  13,
    [220,  20,  60], # pedestrian    =   4,
    [  0,   0, 142], # vehicle       =  10,
]

UNUSED_PALETTE = [
    [ 55,  90,  80], # other         =   3,
    [  0,   0,   0], # unlabeled     =   0,
    [157, 234,  50], # road line     =   6,
    
    [150, 100, 100], # bridge        =  15,
    [110, 190, 160], # static        =  19,
    [170, 120,  50], # dynamic       =  20,
    [ 81,   0,  81], # ground        =  14,
    [180, 165, 180], # guard rail    =  17,
    [230, 150, 140], # rail track    =  16,
    
    [ 45,  60, 150], # water         =  21,
    [236, 236, 236], # general anomaly = 23,
]

OTHERS = [44, 43, 245]
ANOMALY = [236, 236, 236]
import numpy as np
from PIL import Image
from IPython import embed
def convert_rgb_to_mask(rgb):
    """Convert H * W * 3 color map to H * W mask

    Args:
        rgb (torch.Tensor): Color map, torch.Tensor with shape of H * W * 3
        Others: len(PALETTE)
        anomaly: 255
    Returns:
        torch.Tensor: Semantic mask, torch.Tensor with shape of H * W
    """
    assert len(rgb.shape) == 3 and rgb.shape[2] == 3, "Input tensor must be a tensor of H * W * 3"
    h, w = rgb.shape[:2]
    im = rgb.reshape((-1, 3))
    result = len(PALETTE) * np.ones(im.shape[0], dtype=np.uint8)
    for cateid, catergb in enumerate(PALETTE):
        if isinstance(catergb[0], list):
            val = np.array(catergb[0], dtype=np.uint8)
            mask = (im == val).all(axis=1)
            result[mask] = cateid
            val = np.array(catergb[1], dtype=np.uint8)
            mask = (im == val).all(axis=1)
            result[mask] = cateid
        else:
            val = np.array(catergb, dtype=np.uint8)
            mask = (im == val).all(axis=1)
            result[mask] = cateid
    result[(im == np.array([236,236,236],dtype=np.uint8)).all(axis=1)] = 255

    result = result.reshape((h, w))
    return result

def convert_mask_to_rgb(mask):
    """Convert H * W mask to H * W * 3 color map

    Args:
        mask (torch.Tensor): Semantic mask, torch.Tensor with shape of H * W
        Anomaly: yellow
        Others: OTHERS
    Returns:
        torch.Tensor: Color map, torch.Tensor with shape of H * W * 3
    """
    assert len(mask.shape) == 2, 'Input tensor must be a tensor of H * W'
    h, w = mask.shape
    # Image.fromarray(np.asarray(mask)).save('test.png')
    im = mask.flatten()
    result = np.zeros((im.shape[0], 3), dtype=np.uint8)
    result[:, 0] = 236
    result[:, 1] = 236
    result[:, 2] = 236
    for cateid, catergb in enumerate(PALETTE):
        val = np.array(catergb[0] if isinstance(catergb[0], list) else catergb, dtype=np.uint8)
        mask = (im == cateid)
        result[mask, :] = val
    result[im == len(PALETTE), :] = np.array(OTHERS)
    result = result.reshape((h, w, 3))
    # Image.fromarray(np.asarray(result)).save('test2.png')
    return result

import sys
import os
from tqdm import tqdm
if __name__ == '__main__':
    base_dir = sys.argv[1]
    for index in tqdm(sorted(os.listdir(f'{base_dir}')), position=1):
        os.makedirs(f'{base_dir}/{index}/mask_v_idx', exist_ok=True)
        os.makedirs(f'{base_dir}/{index}/mask_v_remap', exist_ok=True)
        for img_name in tqdm(sorted(os.listdir(f'{base_dir}/{index}/mask_v')), position=2):
            mask = convert_rgb_to_mask(np.asarray(Image.open(f'{base_dir}/{index}/mask_v/{img_name}')))
            Image.fromarray(mask).save(f'{base_dir}/{index}/mask_v_idx/{img_name}')
            img = convert_mask_to_rgb(mask)
            Image.fromarray(img).save(f'{base_dir}/{index}/mask_v_remap/{img_name}')
        os.makedirs(f'{base_dir}/{index}/mask_x_idx', exist_ok=True)
        os.makedirs(f'{base_dir}/{index}/mask_x_remap', exist_ok=True)
        for img_name in tqdm(sorted(os.listdir(f'{base_dir}/{index}/mask_v')), position=2):
            mask = convert_rgb_to_mask(np.asarray(Image.open(f'{base_dir}/{index}/mask_x/{img_name}')))
            Image.fromarray(mask).save(f'{base_dir}/{index}/mask_x_idx/{img_name}')
            img = convert_mask_to_rgb(mask)
            Image.fromarray(img).save(f'{base_dir}/{index}/mask_x_remap/{img_name}')
        
        