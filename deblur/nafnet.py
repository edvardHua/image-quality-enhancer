# -*- coding: utf-8 -*-
# @Time : 2023/6/30 11:39
# @Author : zihua.zeng
# @File : nafnet.py

import torch
import cv2
import numpy as np
from models.NAFNet_arch import NAFNetLocal
from utils.img_util import img2tensor, tensor2img, imwrite


def get_pretrain_nafnet_gopro():
    # create model
    # network_g:
    # type: NAFNetLocal
    # width: 32
    # enc_blk_nums: [1, 1, 1, 28]
    # middle_blk_num: 1
    # dec_blk_nums: [1, 1, 1, 1]

    model = NAFNetLocal(width=32, enc_blk_nums=[1, 1, 1, 28], middle_blk_num=1, dec_blk_nums=[1, 1, 1, 1])
    model.load_state_dict(
        torch.load("assets/pretrained_model/NAFNet-GoPro-width32.pth", map_location="cpu")['params'], strict=True)
    model.eval()
    return model