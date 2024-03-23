# Rewritten based on MTI-Net by Yuqi Yang
# Original authors: Simon Vandenhende
# Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)

import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiTaskLoss(nn.Module):
    def __init__(self, p, tasks: list, loss_ft: nn.ModuleDict, loss_weights: dict):
        super(MultiTaskLoss, self).__init__()
        assert(set(tasks) == set(loss_ft.keys()))
        # assert(set(tasks) == set(loss_weights.keys()))
        self.p = p
        self.tasks = tasks
        self.loss_ft = loss_ft
        self.loss_weights = loss_weights
        self.cv_weight = loss_weights['load_balancing']

    
    def forward(self, pred, gt, tasks):
        out = {task: self.loss_ft[task](pred[task], gt[task]) for task in tasks}

        out['total'] = torch.sum(torch.stack([self.loss_weights[t] * out[t] for t in tasks]))
        if 'route_1_prob' in pred:
            loss_cv = 0
            assert('route_2_prob' in pred)
            for task_route in pred['route_1_prob']:
                task_route_list = [t_r for t_r in task_route.values()]
                task_route_tensor = torch.cat(task_route_list, dim=0)
                task_route_tensor = torch.mean(task_route_tensor, dim=0).reshape(-1)
                loss = (torch.std(task_route_tensor) / (torch.mean(task_route_tensor))) ** 2
                loss_cv = loss_cv + loss
            for task_route in pred['route_2_prob']:
                task_route_list = [t_r for t_r in task_route.values()]
                task_route_tensor = torch.cat(task_route_list, dim=0)
                task_route_tensor = torch.mean(task_route_tensor, dim=0).reshape(-1)
                loss = (torch.std(task_route_tensor) / (torch.mean(task_route_tensor))) ** 2
                loss_cv = loss_cv + loss
            out['total'] = out['total'] + loss_cv * self.cv_weight
            out['cv'] = loss_cv 
        return out
