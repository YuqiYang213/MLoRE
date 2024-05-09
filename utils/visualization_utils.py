# By Yuqi Yang
# Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)

import matplotlib.pyplot as plt
from PIL import Image
import imageio, os, cv2
import numpy as np
from utils.utils import get_output
import torch.nn.functional as F
import torch
import warnings
from concurrent.futures import ThreadPoolExecutor

def uint82bin(n, count=8):
    """returns the binary of integer n, count refers to amount of bits"""
    return ''.join([str((n >> y) & 1) for y in range(count-1, -1, -1)])

def labelcolormap(N):
    cmap = np.zeros((N, 3), dtype = np.uint8)
    for i in range(N):
        r = 0
        g = 0
        b = 0
        id = i
        for j in range(7):
            str_id = uint82bin(id)
            r = r ^ ( np.uint8(str_id[-1]) << (7-j))
            g = g ^ ( np.uint8(str_id[-2]) << (7-j))
            b = b ^ ( np.uint8(str_id[-3]) << (7-j))
            id = id >> 3
        cmap[i, 0] = r
        cmap[i, 1] = g
        cmap[i, 2] = b
    return cmap

def vis_semseg(p, _semseg):
    _semseg = _semseg.astype(np.uint64)
    if p['train_db_name'] == "NYUD":
        new_cmap = labelcolormap(40)
    elif p['train_db_name'] == "PASCALContext":
        new_cmap = labelcolormap(21)
    _semseg = new_cmap[_semseg]  
    return _semseg

def vis_parts(inp):
    new_cmap = labelcolormap(7)
    inp = inp.astype(np.uint64)
    inp = new_cmap[inp]
    return inp

def visulization_for_gt(p, sample, save_dir, task):
    inputs, meta = sample['image'], sample['meta']
    im_name = meta['img_name']
    pred = sample[task]

    arr = pred
    if task == 'semseg':
        arr[arr == 255] = 0
        arr = vis_semseg(p, arr)
    elif task == 'sal':
        arr = arr * 255
        arr = arr.astype(np.uint64)
    elif task == 'edge':
        arr = arr * 255
        arr = arr.astype(np.uint64)
    elif task == 'human_parts':
        arr[arr == 255] = 0
        arr = vis_parts(arr)
    elif task == 'normals':
        arr = (arr + 1) / 2 * 255
        arr = arr.astype(np.uint64)
    elif task == 'depth':
        arr = arr.squeeze()
        arr = (arr - arr.min()) / (arr.max() - arr.min()) * 255
        arr_colored = cv2.applyColorMap((arr).astype(np.uint8), cv2.COLORMAP_JET)
        filepath = os.path.join(save_dir, '{}_{}.png'.format(im_name, task))
        cv2.imwrite(filepath, arr_colored)
        return

    if arr.shape[2] == 1:
        arr = arr.squeeze(2)
    arr_uint8 = arr
    if arr_uint8.ndim == 3:
        arr_uint8 = arr_uint8[:, :, [2, 1, 0]] # Convert RGB to BGR for OpenCV
    # else:
    #     arr_uint8 = cv2.applyColorMap(arr_uint8, cv2.COLORMAP_JET)
    filename = '{}_{}.png'.format(im_name, task)
    filepath = os.path.join(save_dir, filename)
    cv2.imwrite(filepath, arr_uint8) 

@torch.no_grad()
def vis_pred_for_one_task(p, sample, output, save_dir, task):
    inputs, meta = sample['image'], sample['meta']
    bs = int(inputs.size()[0])

    warnings.warn('Warning: We assume all the images have the same size!!!')
    im_height = meta['img_size'][0][0]
    im_width = meta['img_size'][0][1]    
    if task == 'semseg':
        output_task = F.interpolate(output[task], (im_height, im_width), mode='bilinear')
        # During visualization, here we always use the train class to draw prediction (totally 19)
        output_task = get_output(output_task, task).cpu().data.numpy()
    else:
        output_task = F.interpolate(output[task], (im_height, im_width), mode='bilinear')
        output_task = get_output(output_task, task).cpu().data.numpy()

    if False: 
        # Serial
        for jj in range(int(inputs.size()[0])):
            im_name = meta['img_name'][jj]
            pred = output_task[jj] # (H, W) or (H, W, C)

            # visualize result 
            arr = pred # (H, W, (C))
            if task == 'semseg':
                arr = vis_semseg(p, arr)
            elif task == 'sal':
                pass
            elif task == 'edge':
                pass
            elif task == 'human_parts':
                arr = vis_parts(arr)
            elif task == 'normals':
                pass
            elif task == 'depth':
                arr = arr.squeeze()
                plt.imsave(os.path.join(save_dir, '{}_{}.png'.format(im_name, task)), arr, cmap='jet')
                continue
            arr_uint8 = arr.astype(np.uint8)
            filename = '{}_{}.png'.format(im_name, task)
            filepath = os.path.join(save_dir, filename)
            plt.imsave(filepath, arr_uint8)
    else:
        # parallel
        def save_image(meta, output_task, save_dir, task, idx):
            im_name = meta['img_name'][idx]
            pred = output_task[idx]

            arr = pred
            if task == 'semseg':
                arr = vis_semseg(p, arr)
            elif task == 'sal':
                pass
            elif task == 'edge':
                pass
            elif task == 'human_parts':
                arr = vis_parts(arr)
            elif task == 'normals':
                pass
            elif task == 'depth':
                arr = arr.squeeze()
                arr = (arr - arr.min()) / (arr.max() - arr.min()) * 255
                arr_colored = cv2.applyColorMap((arr).astype(np.uint8), cv2.COLORMAP_JET)
                filepath = os.path.join(save_dir, '{}_{}.png'.format(im_name, task))
                cv2.imwrite(filepath, arr_colored)
                return

            arr_uint8 = arr.astype(np.uint8)
            if arr_uint8.ndim == 3:
                arr_uint8 = arr_uint8[:, :, [2, 1, 0]] # Convert RGB to BGR for OpenCV
            # else:
            #     arr_uint8 = cv2.applyColorMap(arr_uint8, cv2.COLORMAP_JET)
            filename = '{}_{}.png'.format(im_name, task)
            filepath = os.path.join(save_dir, filename)
            cv2.imwrite(filepath, arr_uint8) 

        def save_images_in_parallel(meta, output_task, save_dir, task):
            with ThreadPoolExecutor() as executor:
                futures = [executor.submit(save_image, meta, output_task, save_dir, task, idx) for idx in range(int(inputs.size()[0]))]
                _ = [future.result() for future in futures]

        save_images_in_parallel(meta, output_task, save_dir, task)
    return
