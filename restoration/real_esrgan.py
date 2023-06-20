# -*- coding: utf-8 -*-
# @Time : 2023/5/5 16:28
# @Author : zihua.zeng
# @File : real_esrgan.py


import os
import torch
from models.rrdbnet_arch import RRDBNet


def _load_model(weight_path, fp32=False):
    model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
    loadnet = torch.load(weight_path, map_location=torch.device('cpu'))
    model.load_state_dict(loadnet['params_ema'], strict=True)
    model.eval()
    if not fp32:
        model = model.half()
    return model


def get_pretrained_real_esrgan_x4():
    return _load_model("assets/pretrained_model/RealESRGAN_x4plus.pth")


def get_pretrained_real_esrnet_x4():
    return _load_model("assets/pretrained_model/RealESRNet_x4plus.pth")


if __name__ == '__main__':
    pass
