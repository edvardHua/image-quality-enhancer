# -*- coding: utf-8 -*-
# @Time : 2023/5/5 16:21
# @Author : zihua.zeng
# @File : nafnet.py


import torch
import cv2
import numpy as np
from models.NAFNet_arch import NAFNet
from utils.img_util import img2tensor, tensor2img, imwrite


def get_pretrain_nafnet():
    # create model
    # network_g:
    # type: NAFNet
    # width: 32
    # enc_blk_nums: [2, 2, 4, 8]
    # middle_blk_num: 12
    # dec_blk_nums: [2, 2, 2, 2]

    model = NAFNet(width=32, enc_blk_nums=[2, 2, 4, 8], middle_blk_num=12, dec_blk_nums=[2, 2, 2, 2])
    model.load_state_dict(
        torch.load("assets/pretrained_model/NAFNet-SIDD-width32.pth", map_location="cpu")['params'], strict=True)
    model.eval()
    return model


if __name__ == '__main__':
    pass
