# Yuqi Yang
# Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)

import torch.nn as nn
import torch.nn.functional as F

INTERPOLATE_MODE = 'bilinear'

class MLoREWrapper(nn.Module):
    def __init__(self, p, backbone, heads, aux_heads=None):
        super(MLoREWrapper, self).__init__()
        self.tasks = p.TASKS.NAMES

        self.backbone = backbone
        self.heads = heads 



    def forward(self, x,  need_info=False):
        img_size = x.size()[-2:]
        out = {}

        target_size = img_size

        task_features, info = self.backbone(x) 
        
        # Generate predictions
        out = {}
        for t in self.tasks:
            if t in task_features:
                _task_fea = task_features[t]
                out[t] = F.interpolate(self.heads[t](_task_fea), target_size, mode=INTERPOLATE_MODE)
        for key in info.keys():
            if 'route' in key:
                out[key] = info[key]
        if need_info:
            return out, info
        else:
            return out
