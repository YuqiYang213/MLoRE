# Yuqi Yang
# Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)
# Based on Vision Transformer (ViT) in PyTorch by Ross Wightman

INTERPOLATE_MODE = 'bilinear'
import math
import logging
from functools import partial
from collections import OrderedDict
from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD, IMAGENET_INCEPTION_MEAN, IMAGENET_INCEPTION_STD
from timm.models.helpers import build_model_with_cfg, named_apply, adapt_input_conv
from timm.models.layers import PatchEmbed, Mlp, DropPath, trunc_normal_, lecun_normal_

import numpy as np
from einops import rearrange as o_rearrange
def rearrange(*args, **kwargs):
    return o_rearrange(*args, **kwargs).contiguous()

BatchNorm2d = nn.BatchNorm2d
_logger = logging.getLogger(__name__)

def _cfg(url='', **kwargs):
    return {
        'url': url,
        'input_size': (3, 224, 224), 'pool_size': None,
        'crop_pct': .9, 'interpolation': 'bicubic', 'fixed_input_size': True,
        'mean': IMAGENET_INCEPTION_MEAN, 'std': IMAGENET_INCEPTION_STD,
        'first_conv': 'patch_embed.proj', 'classifier': 'head',
        # 'num_classes': 1000,
        **kwargs
    }

def sep_prompt(x, prompt_length):
    prompt = x[:, :prompt_length, :]
    x = x[:, prompt_length:, :]
    return prompt, x

default_cfgs = {
    # patch models (weights from official Google JAX impl)
    'vit_tiny_patch16_224': _cfg(
        url='https://storage.googleapis.com/vit_models/augreg/'
            'Ti_16-i21k-300ep-lr_0.001-aug_none-wd_0.03-do_0.0-sd_0.0--imagenet2012-steps_20k-lr_0.03-res_224.npz'),
    'vit_tiny_patch16_384': _cfg(
        url='https://storage.googleapis.com/vit_models/augreg/'
            'Ti_16-i21k-300ep-lr_0.001-aug_none-wd_0.03-do_0.0-sd_0.0--imagenet2012-steps_20k-lr_0.03-res_384.npz',
        input_size=(3, 384, 384), crop_pct=1.0),
    'vit_small_patch32_224': _cfg(
        url='https://storage.googleapis.com/vit_models/augreg/'
            'S_32-i21k-300ep-lr_0.001-aug_light1-wd_0.03-do_0.0-sd_0.0--imagenet2012-steps_20k-lr_0.03-res_224.npz'),
    'vit_small_patch32_384': _cfg(
        url='https://storage.googleapis.com/vit_models/augreg/'
            'S_32-i21k-300ep-lr_0.001-aug_light1-wd_0.03-do_0.0-sd_0.0--imagenet2012-steps_20k-lr_0.03-res_384.npz',
        input_size=(3, 384, 384), crop_pct=1.0),
    'vit_small_patch16_224': _cfg(
        url='https://storage.googleapis.com/vit_models/augreg/'
            'S_16-i21k-300ep-lr_0.001-aug_light1-wd_0.03-do_0.0-sd_0.0--imagenet2012-steps_20k-lr_0.03-res_224.npz'),
    'vit_small_patch16_384': _cfg(
        url='https://storage.googleapis.com/vit_models/augreg/'
            'S_16-i21k-300ep-lr_0.001-aug_light1-wd_0.03-do_0.0-sd_0.0--imagenet2012-steps_20k-lr_0.03-res_384.npz',
        input_size=(3, 384, 384), crop_pct=1.0),
    'vit_base_patch32_224': _cfg(
        url='https://storage.googleapis.com/vit_models/augreg/'
            'B_32-i21k-300ep-lr_0.001-aug_medium1-wd_0.03-do_0.0-sd_0.0--imagenet2012-steps_20k-lr_0.03-res_224.npz'),
    'vit_base_patch32_384': _cfg(
        url='https://storage.googleapis.com/vit_models/augreg/'
            'B_32-i21k-300ep-lr_0.001-aug_light1-wd_0.1-do_0.0-sd_0.0--imagenet2012-steps_20k-lr_0.03-res_384.npz',
        input_size=(3, 384, 384), crop_pct=1.0),
    'vit_base_patch16_224': _cfg(
        url='https://storage.googleapis.com/vit_models/augreg/'
            'B_16-i21k-300ep-lr_0.001-aug_medium1-wd_0.1-do_0.0-sd_0.0--imagenet2012-steps_20k-lr_0.01-res_224.npz'),
    'vit_base_patch16_384': _cfg(
        url='https://storage.googleapis.com/vit_models/augreg/'
            'B_16-i21k-300ep-lr_0.001-aug_medium1-wd_0.1-do_0.0-sd_0.0--imagenet2012-steps_20k-lr_0.01-res_384.npz',
        input_size=(3, 384, 384), crop_pct=1.0),
    'vit_large_patch32_224': _cfg(
        url='',  # no official model weights for this combo, only for in21k
        ),
    'vit_large_patch32_384': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_large_p32_384-9b920ba8.pth',
        input_size=(3, 384, 384), crop_pct=1.0),
    'vit_large_patch16_224': _cfg(
        url='https://storage.googleapis.com/vit_models/augreg/'
            'L_16-i21k-300ep-lr_0.001-aug_medium1-wd_0.1-do_0.1-sd_0.1--imagenet2012-steps_20k-lr_0.01-res_224.npz'),
    'vit_large_patch16_384': _cfg(
        url='https://storage.googleapis.com/vit_models/augreg/'
            'L_16-i21k-300ep-lr_0.001-aug_medium1-wd_0.1-do_0.1-sd_0.1--imagenet2012-steps_20k-lr_0.01-res_384.npz',
        input_size=(3, 384, 384), crop_pct=1.0),

    # patch models, imagenet21k (weights from official Google JAX impl)
    'vit_tiny_patch16_224_in21k': _cfg(
        url='https://storage.googleapis.com/vit_models/augreg/Ti_16-i21k-300ep-lr_0.001-aug_none-wd_0.03-do_0.0-sd_0.0.npz',
        num_classes=21843),
    'vit_small_patch32_224_in21k': _cfg(
        url='https://storage.googleapis.com/vit_models/augreg/S_32-i21k-300ep-lr_0.001-aug_light1-wd_0.03-do_0.0-sd_0.0.npz',
        num_classes=21843),
    'vit_small_patch16_224_in21k': _cfg(
        url='https://storage.googleapis.com/vit_models/augreg/S_16-i21k-300ep-lr_0.001-aug_light1-wd_0.03-do_0.0-sd_0.0.npz',
        num_classes=21843),
    'vit_base_patch32_224_in21k': _cfg(
        url='https://storage.googleapis.com/vit_models/augreg/B_32-i21k-300ep-lr_0.001-aug_medium1-wd_0.03-do_0.0-sd_0.0.npz',
        num_classes=21843),
    'vit_base_patch16_224_in21k': _cfg(
        url='https://storage.googleapis.com/vit_models/augreg/B_16-i21k-300ep-lr_0.001-aug_medium1-wd_0.1-do_0.0-sd_0.0.npz',
        num_classes=21843),
    'vit_large_patch32_224_in21k': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_large_patch32_224_in21k-9046d2e7.pth',
        num_classes=21843),
    'vit_large_patch16_224_in21k': _cfg(
        url='https://storage.googleapis.com/vit_models/augreg/L_16-i21k-300ep-lr_0.001-aug_medium1-wd_0.1-do_0.1-sd_0.1.npz',
        num_classes=21843),
    'vit_huge_patch14_224_in21k': _cfg(
        url='https://storage.googleapis.com/vit_models/imagenet21k/ViT-H_14.npz',
        hf_hub='timm/vit_huge_patch14_224_in21k',
        num_classes=21843),

    # SAM trained models (https://arxiv.org/abs/2106.01548)
    'vit_base_patch32_sam_224': _cfg(
        url='https://storage.googleapis.com/vit_models/sam/ViT-B_32.npz'),
    'vit_base_patch16_sam_224': _cfg(
        url='https://storage.googleapis.com/vit_models/sam/ViT-B_16.npz'),

    # deit models (FB weights)
    'deit_tiny_patch16_224': _cfg(
        url='https://dl.fbaipublicfiles.com/deit/deit_tiny_patch16_224-a1311bcf.pth',
        mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
    'deit_small_patch16_224': _cfg(
        url='https://dl.fbaipublicfiles.com/deit/deit_small_patch16_224-cd65a155.pth',
        mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
    'deit_base_patch16_224': _cfg(
        url='https://dl.fbaipublicfiles.com/deit/deit_base_patch16_224-b5f2ef4d.pth',
        mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
    'deit_base_patch16_384': _cfg(
        url='https://dl.fbaipublicfiles.com/deit/deit_base_patch16_384-8de9b5d1.pth',
        mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD, input_size=(3, 384, 384), crop_pct=1.0),
    'deit_tiny_distilled_patch16_224': _cfg(
        url='https://dl.fbaipublicfiles.com/deit/deit_tiny_distilled_patch16_224-b40b3cf7.pth',
        mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD, classifier=('head', 'head_dist')),
    'deit_small_distilled_patch16_224': _cfg(
        url='https://dl.fbaipublicfiles.com/deit/deit_small_distilled_patch16_224-649709d9.pth',
        mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD, classifier=('head', 'head_dist')),
    'deit_base_distilled_patch16_224': _cfg(
        url='https://dl.fbaipublicfiles.com/deit/deit_base_distilled_patch16_224-df68dfff.pth',
        mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD, classifier=('head', 'head_dist')),
    'deit_base_distilled_patch16_384': _cfg(
        url='https://dl.fbaipublicfiles.com/deit/deit_base_distilled_patch16_384-d0272ac0.pth',
        mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD, input_size=(3, 384, 384), crop_pct=1.0,
        classifier=('head', 'head_dist')),

    # ViT ImageNet-21K-P pretraining by MILL
    'vit_base_patch16_224_miil_in21k': _cfg(
        url='https://miil-public-eu.oss-eu-central-1.aliyuncs.com/model-zoo/ImageNet_21K_P/models/timm/vit_base_patch16_224_in21k_miil.pth',
        mean=(0, 0, 0), std=(1, 1, 1), crop_pct=0.875, interpolation='bilinear', num_classes=11221,
    ),
    'vit_base_patch16_224_miil': _cfg(
        url='https://miil-public-eu.oss-eu-central-1.aliyuncs.com/model-zoo/ImageNet_21K_P/models/timm'
            '/vit_base_patch16_224_1k_miil_84_4.pth',
        mean=(0, 0, 0), std=(1, 1, 1), crop_pct=0.875, interpolation='bilinear',
    ),
}


class Attention(nn.Module):
    def __init__(self, chan_nheads, resolution, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.dim = dim
        self.resolution = resolution
        pixel_no = int(resolution[0] * resolution[1])
        self.pixel_no = pixel_no

        self.chan_nheads = chan_nheads
        chan_head_dim = self.pixel_no // self.chan_nheads
        self.chan_scale = chan_head_dim ** -0.5

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   

        raw_spa_attn = (q @ k.transpose(-2, -1))
        attn = raw_spa_attn * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        raw_spa_attn = raw_spa_attn, attn

        x = (attn @ v).transpose(1, 2).reshape(B, N, C) # (B, task_no+1+HxW, C)

        x = self.proj(x)
        x = self.proj_drop(x)
        raw_attn = [raw_spa_attn]

        return x, raw_attn

class LoraBlock(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=1, rank=6):
        super().__init__()
        self.W = nn.Conv2d(in_channels, rank, kernel_size=kernel_size, stride=1, padding=kernel_size//2)
        self.M = nn.Conv2d(rank, out_channels, kernel_size=1, stride=1)

    def init_weights(self):
        nn.init.kaiming_uniform_(self.W.weight, a=math.sqrt(5))
        nn.init.zeros_(self.W.bias)
        nn.init.kaiming_uniform_(self.M.weight, a=math.sqrt(5))
        nn.init.zeros_(self.M.bias)
    
    def forward(self, x):
        x = self.W(x)
        x = self.M(x)
        return x


class Block(nn.Module):

    def __init__(self, chan_nheads, resolution, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(chan_nheads, resolution, dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):

        x_attn, attn_weight = self.attn(self.norm1(x))
        x = x + self.drop_path(x_attn)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x, attn_weight

class SpatialAtt(nn.Module):
    def __init__(self, dim, dim_out, im_size, with_feat):
        super().__init__()
        self.conv1 = nn.Conv2d(dim, dim_out, kernel_size=1)
        self.act = nn.GELU()
        self.ln = nn.LayerNorm(dim_out)
        self.convsp = nn.Linear(im_size, 1)
        self.ln_sp = nn.LayerNorm(dim)
        self.conv2 = nn.Conv2d(dim, dim_out, kernel_size=1)
        self.conv3 = nn.Conv2d(dim_out, dim_out, kernel_size=1)
        self.with_feat = with_feat
        if with_feat:
            self.feat_linear = nn.Conv2d(dim_out *2 , dim_out *2, kernel_size=1)
    
    def forward(self, x, route_feat=None):
        n, _, h, w = x.shape
        feat = self.conv1(x)
        feat = self.ln(feat.reshape(n, -1, h * w).permute(0, 2, 1)).permute(0, 2, 1).reshape(n, -1, h, w)
        feat = self.act(feat)
        feat = self.conv3(feat)

        feat_sp = self.convsp(x.reshape(n, -1, h * w)).reshape(n, 1, -1)
        feat_sp = self.ln_sp(feat_sp).reshape(n, -1, 1, 1)
        feat_sp = self.act(feat_sp)
        feat_sp = self.conv2(feat_sp)
        
        n, c, h, w = feat.shape
        feat = torch.mean(feat.reshape(n, c, h * w), dim=2).reshape(n, c, 1, 1)
        feat = torch.cat([feat, feat_sp], dim=1)

        return feat

class MOEBlock(nn.Module):
    def __init__(self, p, final_embed_dim, im_size, kernel_size=1, with_feat=False):
        super().__init__()
        self.num_lora = len(p.rank_list)
        self.p = p
        self.lora_list_1 = nn.ModuleList()
        rank_list = p.rank_list
        for i in range(self.num_lora):
            self.lora_list_1.append(LoraBlock(final_embed_dim, final_embed_dim, kernel_size=kernel_size, rank=rank_list[i]))
            self.lora_list_1[i].init_weights()
        self.conv1 = nn.ModuleDict()
        self.conv2 = nn.ModuleDict()
        self.conv3 = nn.ModuleDict()
        self.share_conv = nn.Conv2d(final_embed_dim, final_embed_dim, kernel_size=3, padding=1)
        self.bn = nn.ModuleDict()
        self.bn_all = nn.ModuleDict()
        self.activate = nn.GELU()
        for task in self.p.TASKS.NAMES:
            self.conv1[task] = nn.Conv2d(final_embed_dim, final_embed_dim, kernel_size=1)
            self.conv3[task] = nn.Conv2d(final_embed_dim, final_embed_dim, kernel_size=1)
            self.conv2[task] = LoraBlock(final_embed_dim, final_embed_dim, kernel_size=kernel_size, rank=p.spe_rank)

            self.bn[task] = BatchNorm2d(final_embed_dim)
            self.bn_all[task] = BatchNorm2d(final_embed_dim)

        self.router_1 = nn.ModuleDict() 
        self.pre_softmax = p.pre_softmax
        self.desert_k = len(p.rank_list) - p.topk
        for task in self.p.TASKS.NAMES:
            self.router_1[task] = nn.ModuleList()
            self.router_1[task].append(SpatialAtt(final_embed_dim, final_embed_dim // 4, im_size=im_size, with_feat=with_feat))
            self.router_1[task].append(nn.Conv2d(final_embed_dim // 2, self.num_lora * 2 + 1, kernel_size=1))
        
    def forward(self, x, task, route_feat_in=None):
        out_ori = self.conv1[task](x)
        out = out_ori
        n, c, h, w = out.shape
        route_feat = self.router_1[task][0](out, route_feat_in)
        prob_all = self.router_1[task][1](route_feat).unsqueeze(2)
        prob_lora, prob_mix = prob_all[:, :self.num_lora * 2], prob_all[:, self.num_lora * 2:]
        route_1_raw, stdev_1 = prob_lora.chunk(2, dim=1)  # n, 15, 1, 1, 1
        if self.training:
            noise = torch.randn_like(route_1_raw) * stdev_1
        else:
            noise = 0
        if self.pre_softmax:
            route_1_raw = route_1_raw + noise
            route_1_indice = torch.topk(route_1_raw, self.desert_k, dim=1, largest=False)[1]
            for j in range(n):
                for i in range(self.desert_k):
                    route_1_raw[j, route_1_indice[j, i].reshape(-1)] = -1e10
            route_1 = torch.softmax(route_1_raw, dim=1)
        else:
            route_1_raw = torch.softmax(route_1_raw + noise, dim=1)
            route_1_indice = torch.topk(route_1_raw, self.desert_k, dim=1, largest=False)[1]
            route_1 = route_1_raw.clone()
            for j in range(n):
                for i in range(self.desert_k):
                    route_1[j, route_1_indice[j, i].reshape(-1)] = 0
        lora_out_1 = []
        for i in range(self.num_lora):
            lora_out_1.append(self.lora_list_1[i](out).unsqueeze(1)) # n, 1, c, h, w
        lora_out_1 = torch.cat(lora_out_1, dim=1)
        lora_out_1 = torch.sum(lora_out_1 * route_1, dim=1)
        out = self.bn_all[task](lora_out_1) + self.conv2[task](out) * prob_mix[:, 0] + self.share_conv(out.detach())
        out = self.bn[task](out)
        out = self.activate(out)

        out = self.conv3[task](out)
        return out, route_feat, route_1

class MLoRE(nn.Module):
    """ MLoRE built upon ViT
    """

    def __init__(self, p, select_list, img_size=224, patch_size=16, in_chans=3, embed_dim=768, depth=12,
                 num_heads=12, chan_nheads=1, mlp_ratio=4., qkv_bias=True,  
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0., embed_layer=PatchEmbed, norm_layer=None,
                 act_layer=None, weight_init=''):
        """
        Args:
            p (dcit): parameters
            select_list: selected layers for hierarchical prompting
            img_size (int, tuple): input image size
            patch_size (int, tuple): patch size
            in_chans (int): number of input channels
            embed_dim (int): embedding dimension
            depth (int): depth of transformer
            num_heads (int): number of attention heads
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim
            qkv_bias (bool): enable bias for qkv if True
            drop_rate (float): dropout rate
            attn_drop_rate (float): attention dropout rate
            drop_path_rate (float): stochastic depth rate
            embed_layer (nn.Module): patch embedding layer
            norm_layer: (nn.Module): normalization layer
            weight_init: (str): weight init scheme
        """
        super().__init__()
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU

        self.patch_embed = embed_layer(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim)) # one cls token from pretrained weights on ImageNet
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.resolution = [int(img_size[0]/patch_size), int(img_size[1]/patch_size)]
        self.blocks = nn.Sequential(*[
            Block(
                chan_nheads,
                self.resolution,
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop_rate,
                attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, act_layer=act_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)

        self.select_list = select_list
        self.num_layers = 4
        assert len(select_list) == self.num_layers-1 
        task_no = len(p.TASKS.NAMES)
        self.resolution = [int(img_size[0]/patch_size), int(img_size[1]/patch_size)]
        pixel_no = int(self.resolution[0] * self.resolution[1])
        self.pixel_no = pixel_no
        self.p = p

        # multi-task prompt learning
        self.prompt_len = 1
        self.prompts_len = task_no*1

        self.fea_fuse = nn.ModuleList()

        attn_conv_expansion= 1
        prompt_dim = num_heads*1
        tar_dim = 700
        final_embed_dim = p.final_embed_dim
        self.num_lora = 15
        self.MLoRE_1 = nn.ModuleList()
        for i_layer in range(self.num_layers):
            self.fea_fuse.append(nn.Conv2d(embed_dim, final_embed_dim, kernel_size=1, stride=1))
            self.MLoRE_1.append(MOEBlock(p, final_embed_dim, im_size=pixel_no, kernel_size=3,with_feat=False))
        self.MLoRE_2 = nn.ModuleList()
        for il in range(self.num_layers):
            self.MLoRE_2.append(MOEBlock(p, final_embed_dim, im_size=pixel_no, kernel_size=3,with_feat=False))
        self.task_mask = nn.ModuleDict()

        self.q_fuse = nn.ModuleDict()
        
        for task in p.TASKS.NAMES:
            self.q_fuse[task] = nn.Conv2d(final_embed_dim * 4, final_embed_dim, kernel_size=1, stride=1, padding=0)
            self.task_mask[task] = nn.Sequential(
                                                nn.Conv2d(final_embed_dim * 4, final_embed_dim, kernel_size=1),
                                                BatchNorm2d(final_embed_dim), nn.GELU(),  
                                                nn.Conv2d(final_embed_dim, self.num_layers, kernel_size=3, padding=1))
        
        self.classify_route_1 = nn.Conv2d(final_embed_dim // 2, len(p.TASKS.NAMES), kernel_size=1)
        self.classify_route_2 = nn.Conv2d(final_embed_dim // 2, len(p.TASKS.NAMES), kernel_size=1)
        
        # for task in p.TASKS.NAMES:
        #     self.task_mask[task] = nn.Sequential(
        #                                         nn.Conv2d(final_embed_dim * 4, final_embed_dim, kernel_size=1),
        #                                         BatchNorm2d(final_embed_dim), nn.GELU(),  
        #                                         nn.Conv2d(final_embed_dim, self.num_layers, kernel_size=3, padding=1))
        
        
        self.init_weights(weight_init)

    def init_weights(self, mode=''):
        assert mode in ('jax', 'jax_nlhb', 'nlhb', '')
        head_bias = -math.log(self.num_classes) if 'nlhb' in mode else 0.
        trunc_normal_(self.pos_embed, std=.02)
        if mode.startswith('jax'):
            # leave cls token as zeros to match jax impl
            named_apply(partial(_init_vit_weights, head_bias=head_bias, jax_impl=True), self)
        else:
            self.apply(_init_vit_weights)

    def _init_weights(self, m):
        # this fn left here for compat with downstream users
        _init_vit_weights(m)

    @torch.jit.ignore()
    def load_pretrained(self, checkpoint_path, prefix=''):
        _load_weights(self, checkpoint_path, prefix)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token', 'dist_token'}

    def forward(self, x, eval=True):
        x = self.patch_embed(x)
        x = self.pos_drop(x + self.pos_embed[:, 1:]) 


        # multi-scale backbone feature
        all_tasks = self.p.TASKS.NAMES
        out_feat = {task: [] for task in all_tasks}
        last_feat = {task:0 for task in all_tasks}
        out_mask = {task: 0 for task in all_tasks}
        info = {} # pass information through the pipeline

        x_q_list = {task:[] for task in self.p.TASKS.NAMES}
        route_feat_1 = {task: None for task in all_tasks}
        route_feat_2 = {task: None for task in all_tasks}
        route_prob_1 = [{task:None for task in all_tasks} for i in range(self.num_layers)]
        route_prob_2 = [{task:None for task in all_tasks} for i in range(self.num_layers)]
        for idx, blk in enumerate(self.blocks):
            x, attn_weight = blk(x)
            if idx + 1 in self.select_list: 
                # extract task-specific feature at this layer
                il = np.sum(idx >= (np.array(self.select_list) - 1)) - 1 # [0,1,2]
                _cur_task_fea, x_q, info, route_feat_1, route_prob_1[il] = self.cal_task_feature(x, attn_weight, il, info, route_feat_1)
                for task in all_tasks:
                    _cur_task_fea_now, route_feat_2[task], route_prob_2[il][task] = self.MLoRE_2[il](_cur_task_fea[task], task, route_feat_2[task])
                    x_q_list[task].append(x_q[task])
                    out_feat[task].append(_cur_task_fea_now)
            
        x = self.norm(x)

        # extract task-specific feature at the last layer
        il=self.num_layers-1
        _cur_task_fea, x_q, info, route_feat_1, route_prob_1[il] = self.cal_task_feature(x, attn_weight, il, info, route_feat_1)
        
        for task in all_tasks:
            _cur_task_fea_now, route_feat_2[task], route_prob_2[il][task] = self.MLoRE_2[il](_cur_task_fea[task], task, route_feat_2[task])
            
            x_q_list[task].append(x_q[task])
            out_feat[task].append(_cur_task_fea_now)

            mask_now = self.task_mask[task](torch.cat(out_feat[task], dim=1))
            mask_now = torch.softmax(mask_now, dim=1)
            for il in range(self.num_layers):
                last_feat[task] = last_feat[task] + mask_now[:, il].unsqueeze(1) * out_feat[task][il]
        info['route_1_prob'] = route_prob_1
        info['route_2_prob'] = route_prob_2
        for task in last_feat.keys():
            last_feat[task] = F.interpolate(last_feat[task], scale_factor=4, mode=INTERPOLATE_MODE)
        return last_feat, info

    def cal_task_feature(self, x, attn_weight, il, info, route_feat):
        ''' Calculate task feature at this layer
        '''
        combined_fea = rearrange(x, 'b (h w) c -> b c h w', h=self.resolution[0], w=self.resolution[1])

        combined_fea = self.fea_fuse[il](combined_fea)

        x_q = {task:0 for task in self.p.TASKS.NAMES}
        route_feat_out = {task:0 for task in self.p.TASKS.NAMES}
        route_prob_out = {task:0 for task in self.p.TASKS.NAMES}
        for task in self.p.TASKS.NAMES:
            x_q[task], route_feat_out[task], route_prob_out[task] = self.MLoRE_1[il](combined_fea, task, route_feat[task])

        combined_fea = x_q

        return combined_fea, x_q, info, route_feat_out, route_prob_out


def _init_vit_weights(module: nn.Module, name: str = '', head_bias: float = 0., jax_impl: bool = False):
    """ ViT weight initialization
    * When called without n, head_bias, jax_impl args it will behave exactly the same
      as my original init for compatibility with prev hparam / downstream use cases (ie DeiT).
    * When called w/ valid n (module name) and jax_impl=True, will (hopefully) match JAX impl
    """
    if isinstance(module, nn.Linear):
        if name.startswith('head'):
            nn.init.zeros_(module.weight)
            nn.init.constant_(module.bias, head_bias)
        elif name.startswith('pre_logits'):
            lecun_normal_(module.weight)
            nn.init.zeros_(module.bias)
        else:
            if jax_impl:
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    if 'mlp' in name:
                        nn.init.normal_(module.bias, std=1e-6)
                    else:
                        nn.init.zeros_(module.bias)
            else:
                trunc_normal_(module.weight, std=.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    elif jax_impl and isinstance(module, nn.Conv2d):
        # NOTE conv was left to pytorch default in my original init
        lecun_normal_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, (nn.LayerNorm, nn.GroupNorm, nn.BatchNorm2d)):
        nn.init.zeros_(module.bias)
        nn.init.ones_(module.weight)


@torch.no_grad()
def _load_weights(model, checkpoint_path: str, prefix: str = ''):
    """ Load weights from .npz checkpoints for official Google Brain Flax implementation
    """
    import numpy as np

    def _n2p(w, t=True):
        if w.ndim == 4 and w.shape[0] == w.shape[1] == w.shape[2] == 1:
            w = w.flatten()
        if t:
            if w.ndim == 4:
                w = w.transpose([3, 2, 0, 1])
            elif w.ndim == 3:
                w = w.transpose([2, 0, 1])
            elif w.ndim == 2:
                w = w.transpose([1, 0])
        return torch.from_numpy(w)

    w = np.load(checkpoint_path)
    if not prefix and 'opt/target/embedding/kernel' in w:
        prefix = 'opt/target/'

    if hasattr(model.patch_embed, 'backbone'):
        # hybrid
        backbone = model.patch_embed.backbone
        stem_only = not hasattr(backbone, 'stem')
        stem = backbone if stem_only else backbone.stem
        stem.conv.weight.copy_(adapt_input_conv(stem.conv.weight.shape[1], _n2p(w[f'{prefix}conv_root/kernel'])))
        stem.norm.weight.copy_(_n2p(w[f'{prefix}gn_root/scale']))
        stem.norm.bias.copy_(_n2p(w[f'{prefix}gn_root/bias']))
        if not stem_only:
            for i, stage in enumerate(backbone.stages):
                for j, block in enumerate(stage.blocks):
                    bp = f'{prefix}block{i + 1}/unit{j + 1}/'
                    for r in range(3):
                        getattr(block, f'conv{r + 1}').weight.copy_(_n2p(w[f'{bp}conv{r + 1}/kernel']))
                        getattr(block, f'norm{r + 1}').weight.copy_(_n2p(w[f'{bp}gn{r + 1}/scale']))
                        getattr(block, f'norm{r + 1}').bias.copy_(_n2p(w[f'{bp}gn{r + 1}/bias']))
                    if block.downsample is not None:
                        block.downsample.conv.weight.copy_(_n2p(w[f'{bp}conv_proj/kernel']))
                        block.downsample.norm.weight.copy_(_n2p(w[f'{bp}gn_proj/scale']))
                        block.downsample.norm.bias.copy_(_n2p(w[f'{bp}gn_proj/bias']))
        embed_conv_w = _n2p(w[f'{prefix}embedding/kernel'])
    else:
        embed_conv_w = adapt_input_conv(
            model.patch_embed.proj.weight.shape[1], _n2p(w[f'{prefix}embedding/kernel']))
    model.patch_embed.proj.weight.copy_(embed_conv_w)
    model.patch_embed.proj.bias.copy_(_n2p(w[f'{prefix}embedding/bias']))
    pos_embed_w = _n2p(w[f'{prefix}Transformer/posembed_input/pos_embedding'], t=False)
    if pos_embed_w.shape != model.pos_embed.shape:
        pos_embed_w = resize_pos_embed(  # resize pos embedding when different size from pretrained weights
            pos_embed_w, model.pos_embed, getattr(model, 'num_tokens', 1), model.patch_embed.grid_size)
    model.pos_embed.copy_(pos_embed_w)
    model.norm.weight.copy_(_n2p(w[f'{prefix}Transformer/encoder_norm/scale']))
    model.norm.bias.copy_(_n2p(w[f'{prefix}Transformer/encoder_norm/bias']))
    for i, block in enumerate(model.blocks.children()):
        block_prefix = f'{prefix}Transformer/encoderblock_{i}/'
        mha_prefix = block_prefix + 'MultiHeadDotProductAttention_1/'
        block.norm1.weight.copy_(_n2p(w[f'{block_prefix}LayerNorm_0/scale']))
        block.norm1.bias.copy_(_n2p(w[f'{block_prefix}LayerNorm_0/bias']))
        block.attn.qkv.weight.copy_(torch.cat([
            _n2p(w[f'{mha_prefix}{n}/kernel'], t=False).flatten(1).T for n in ('query', 'key', 'value')]))
        block.attn.qkv.bias.copy_(torch.cat([
            _n2p(w[f'{mha_prefix}{n}/bias'], t=False).reshape(-1) for n in ('query', 'key', 'value')]))
        block.attn.proj.weight.copy_(_n2p(w[f'{mha_prefix}out/kernel']).flatten(1))
        block.attn.proj.bias.copy_(_n2p(w[f'{mha_prefix}out/bias']))
        for r in range(2):
            getattr(block.mlp, f'fc{r + 1}').weight.copy_(_n2p(w[f'{block_prefix}MlpBlock_3/Dense_{r}/kernel']))
            getattr(block.mlp, f'fc{r + 1}').bias.copy_(_n2p(w[f'{block_prefix}MlpBlock_3/Dense_{r}/bias']))
        block.norm2.weight.copy_(_n2p(w[f'{block_prefix}LayerNorm_2/scale']))
        block.norm2.bias.copy_(_n2p(w[f'{block_prefix}LayerNorm_2/bias']))


def resize_pos_embed(posemb, posemb_new, num_tokens=1, gs_new=()):
    # Rescale the grid of position embeddings when loading from state_dict. Adapted from
    # https://github.com/google-research/vision_transformer/blob/00883dd691c63a6830751563748663526e811cee/vit_jax/checkpoint.py#L224
    _logger.info('Resized position embedding: %s to %s', posemb.shape, posemb_new.shape)
    ntok_new = posemb_new.shape[1]
    if num_tokens:
        posemb_tok, posemb_grid = posemb[:, :num_tokens], posemb[0, num_tokens:]
        ntok_new -= num_tokens
    else:
        posemb_tok, posemb_grid = posemb[:, :0], posemb[0]
    gs_old = int(math.sqrt(len(posemb_grid)))
    if not len(gs_new):  # backwards compatibility
        gs_new = [int(math.sqrt(ntok_new))] * 2
    assert len(gs_new) >= 2
    _logger.info('Position embedding grid-size from %s to %s', [gs_old, gs_old], gs_new)
    posemb_grid = posemb_grid.reshape(1, gs_old, gs_old, -1).permute(0, 3, 1, 2)
    posemb_grid = F.interpolate(posemb_grid, size=gs_new, mode='bicubic', align_corners=False)
    posemb_grid = posemb_grid.permute(0, 2, 3, 1).reshape(1, gs_new[0] * gs_new[1], -1)
    posemb = torch.cat([posemb_tok, posemb_grid], dim=1)
    return posemb


def checkpoint_filter_fn(state_dict, model):
    """ convert patch embedding weight from manual patchify + linear proj to conv"""
    out_dict = {}
    if 'model' in state_dict:
        # For deit models
        state_dict = state_dict['model']
    for k, v in state_dict.items():
        if 'patch_embed.proj.weight' in k and len(v.shape) < 4:
            # For old models that I trained prior to conv based patchification
            O, I, H, W = model.patch_embed.proj.weight.shape
            v = v.reshape(O, -1, H, W)
        elif k == 'pos_embed' and v.shape != model.pos_embed.shape:
            # To resize pos embedding when using model at different size from pretrained weights
            v = resize_pos_embed(
                v, model.pos_embed, getattr(model, 'num_tokens', 1), model.patch_embed.grid_size)
        out_dict[k] = v
    return out_dict


def _create_MLoRE(variant, pretrained=False, default_cfg=None, **kwargs):
    default_cfg = default_cfg or default_cfgs[variant]
    if kwargs.get('features_only', None):
        raise RuntimeError('features_only not implemented for Vision Transformer models.')

    # NOTE this extra code to support handling of repr size for in21k pretrained models
    # default_num_classes = default_cfg['num_classes']
    # num_classes = kwargs.get('num_classes', default_num_classes)
    # repr_size = kwargs.pop('representation_size', None)
    # if repr_size is not None and num_classes != default_num_classes:
    #     # Remove representation layer if fine-tuning. This may not always be the desired action,
    #     # but I feel better than doing nothing by default for fine-tuning. Perhaps a better interface?
    #     _logger.warning("Removing representation layer for fine-tuning.")
    #     repr_size = None
    # print('npz' in default_cfg['file'])
    model = build_model_with_cfg(
        MLoRE, variant, pretrained,
        default_cfg=default_cfg,
        # representation_size=repr_size,
        pretrained_filter_fn=checkpoint_filter_fn,
        pretrained_custom_load='npz' in default_cfg['url'],
        **kwargs)
    return model


def MLoRE_vit_large_patch16_384(pretrained=False, **kwargs):
    """ Based on ViT-Large model (ViT-L/16) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-1k weights fine-tuned from in21k @ 384x384, source https://github.com/google-research/vision_transformer.
    """
    model_kwargs = dict(select_list=range(6,24,6), patch_size=16, embed_dim=1024, depth=24, num_heads=16, **kwargs)
    model = _create_MLoRE('vit_large_patch16_384', pretrained=pretrained, **model_kwargs)
    return model

def MLoRE_vit_base_patch16_384(pretrained=False, **kwargs):
    """ Based on ViT-Base model (ViT-B/16) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-1k weights fine-tuned from in21k @ 384x384, source https://github.com/google-research/vision_transformer.
    """
    model_kwargs = dict(select_list=range(3,12,3), patch_size=16, embed_dim=768, depth=12, num_heads=12,  **kwargs)
    model = _create_MLoRE('vit_base_patch16_384', pretrained=pretrained, **model_kwargs)
    return model

def MLoRE_vit_small_patch16_384(pretrained=False, **kwargs):
    """ Based on ViT-Small model (ViT-S/16) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-1k weights fine-tuned from in21k @ 384x384, source https://github.com/google-research/vision_transformer.
    """
    model_kwargs = dict(select_list=range(3,12,3),patch_size=16, embed_dim=384, depth=12, num_heads=6, **kwargs)
    model = _create_MLoRE('vit_small_patch16_384', pretrained=pretrained, **model_kwargs)
    return model


class ConvHead(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()

        self.mt_proj = nn.Sequential(nn.Conv2d(in_channels, in_channels, 3, padding=1), BatchNorm2d(in_channels), nn.GELU())
        trunc_normal_(self.mt_proj[0].weight, std=0.02)

        self.linear_pred = nn.Conv2d(in_channels, num_classes, kernel_size=1)
        nn.init.normal(self.linear_pred.bias, mean=0, std=0.02)

    def forward(self, x):
        return self.linear_pred(self.mt_proj(x))
    
class DEConvHead(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()

        self.mt_proj = nn.Sequential(
            nn.ConvTranspose2d(in_channels, in_channels//2, 2, stride=2, padding=0), BatchNorm2d(in_channels//2), nn.GELU(),
            nn.Conv2d(in_channels//2, in_channels//2, 3, padding=1), BatchNorm2d(in_channels//2), nn.GELU()
            )

        self.linear_pred = nn.Conv2d(in_channels//2, num_classes, kernel_size=1)
        trunc_normal_(self.mt_proj[0].weight, std=0.02)
        trunc_normal_(self.mt_proj[3].weight, std=0.02)
        trunc_normal_(self.linear_pred.weight, std=0.02)

    def forward(self, x):
        return self.linear_pred(self.mt_proj(x))
