# Yuqi Yang
# Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)

from evaluation.evaluate_utils import PerformanceMeter
from tqdm import tqdm
from utils.utils import get_output, mkdir_if_missing
from evaluation.evaluate_utils import save_model_pred_for_one_task
import torch
import os

@torch.no_grad()
def test_phase(p, test_loader, model, criterion, epoch):
    all_tasks = [t for t in p.TASKS.NAMES]
    two_d_tasks = [t for t in p.TASKS.NAMES]
    performance_meter = PerformanceMeter(p, two_d_tasks)

    model.eval()

    tasks_to_save = []
    # if 'depth' in all_tasks:
    #     tasks_to_save.append('depth')
    # if 'normals' in all_tasks:
    #     tasks_to_save.append('normals')
    # if 'edge' in all_tasks:
    #     tasks_to_save.append('edge')

    save_dirs = {task: os.path.join(p['save_dir'], task) for task in tasks_to_save}
    for save_dir in save_dirs.values():
        mkdir_if_missing(save_dir)

    route_1_task_sum = []
    route_2_task_sum = []
    for i in range(4):
        route_1_task_sum.append(0)
        route_2_task_sum.append(0)
    route_1_task = []
    route_2_task = []
    for i in range(4):
        route_1_task.append({task:0 for task in p.TASKS.NAMES})
        route_2_task.append({task:0 for task in p.TASKS.NAMES})
    
    for i, batch in enumerate(tqdm(test_loader)):
        # Forward pass
        with torch.no_grad():
            images = batch['image'].cuda(non_blocking=True)
            targets = {task: batch[task].cuda(non_blocking=True) for task in two_d_tasks}

            output = model.module(images) # to make ddp happy
        
            # Measure loss and performance
            performance_meter.update({t: get_output(output[t], t) for t in two_d_tasks}, 
                                    {t: targets[t] for t in two_d_tasks})

            for task in tasks_to_save:
                save_model_pred_for_one_task(p, i, batch, output, save_dirs, task, epoch=epoch)

    eval_results = performance_meter.get_score(verbose = True)

    return eval_results


@torch.no_grad()
def vis_phase(p, test_loader, model, criterion, epoch):
    from utils.visualization_utils import vis_pred_for_one_task
    # Originally designed for visualization on cityscapes-3D
    model.eval()

    tasks_to_save = ['semseg', 'depth']

    save_dirs = {task: os.path.join(p['save_dir'], 'vis', task) for task in tasks_to_save}
    for save_dir in save_dirs.values():
        mkdir_if_missing(save_dir)
    
    for i, batch in enumerate(tqdm(test_loader)):
        # Forward pass
        images = batch['image'].cuda(non_blocking=True)
        output = model.module(images) # to make ddp happy
    
        for task in tasks_to_save:
            vis_pred_for_one_task(p, batch, output, save_dirs[task], task)
        del batch, output, images


    return
