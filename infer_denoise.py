# -*- coding: utf-8 -*-
# @Time : 2023/6/5 11:49
# @Author : zihua.zeng
# @File : infer_denoise.py

import cv2
import torch
import numpy as np
from denoise.nafnet import get_pretrain_nafnet
from utils.img_util import img2tensor, tensor2img, imwrite


def test_pipeline():
    model = get_pretrain_nafnet()
    ori_img = cv2.imread("assets/test_denoise.png")
    ori_img = ori_img.astype(np.float32) / 255.
    ori_img_tensor = img2tensor(ori_img, bgr2rgb=True, float32=True)
    out = model(ori_img_tensor.unsqueeze(0))
    out_img = tensor2img(out.squeeze(0))
    imwrite(out_img, "denoise_img.png")


if __name__ == '__main__':
    test_pipeline()
    pass
