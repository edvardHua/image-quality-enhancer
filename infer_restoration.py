# -*- coding: utf-8 -*-
# @Time : 2023/6/9 16:25
# @Author : zihua.zeng
# @File : infer_restoration.py

import cv2
import numpy as np
import torch

from restoration.nafnet_reds import get_pretrain_nafnet_reds
from restoration.real_esrgan import get_pretrained_real_esrgan_x4, get_pretrained_real_esrnet_x4
from utils.img_util import img2tensor, tensor2img, imwrite

device = torch.device("cuda:0")


def test_jpg_compress_by_nafnet(input_image, output_image):
    model = get_pretrain_nafnet_reds()
    model = model.to(device)
    ori_img = cv2.imread(input_image)
    ori_img = ori_img.astype(np.float32) / 255.
    ori_img_tensor = img2tensor(ori_img, bgr2rgb=True, float32=True)
    out = model(ori_img_tensor.unsqueeze(0))
    out_img = tensor2img(out.squeeze(0))
    imwrite(out_img, output_image)


def test_realesrgan_x4(input_image, output_image):
    model = get_pretrained_real_esrgan_x4()
    model = model.to(device)
    ori_img = cv2.imread(input_image)
    ori_img = ori_img.astype(np.float32) / 255.
    ori_img_tensor = img2tensor(ori_img, bgr2rgb=True, float32=True)
    out = model((ori_img_tensor.unsqueeze(0).half()).to(device))
    out_img = tensor2img(out.squeeze(0))
    imwrite(out_img, output_image)


if __name__ == '__main__':
    image_path = "assets/id-11134201-23030-4fulp55fzbov7a.jpg"
    out_path = "camp_out.jpg"
    # test_jpg_compress_by_nafnet(image_path, out_path)
    test_realesrgan_x4(image_path, out_path)
    pass
