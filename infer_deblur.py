# -*- coding: utf-8 -*-
# @Time : 2023/6/30 11:40
# @Author : zihua.zeng
# @File : infer_deblur.py


import os
import cv2
import numpy as np
import torch

from tqdm import tqdm
from deblur.nafnet import get_pretrain_nafnet_gopro
from restoration.real_esrgan import get_pretrained_real_esrgan_x4, get_pretrained_real_esrnet_x4
from utils.img_util import img2tensor, tensor2img, imwrite

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")


def test_deblur_by_nafnet(input_image, output_image):
    model = get_pretrain_nafnet_gopro()
    model = model.to(device)
    ori_img = cv2.imread(input_image)
    ori_img = ori_img.astype(np.float32) / 255.
    ori_img_tensor = img2tensor(ori_img, bgr2rgb=True, float32=True)
    ori_img_tensor = ori_img_tensor.to(device)
    out = model(ori_img_tensor.unsqueeze(0))
    out_img = tensor2img(out.squeeze(0))
    imwrite(out_img, output_image)


def unit_test1():
    """
    单图
    :return:
    """
    image_path = "assets/id-11134201-23030-4fulp55fzbov7a.jpg"
    out_path = "camp_out.jpg"
    # test_jpg_compress_by_nafnet(image_path, out_path)
    test_deblur_by_nafnet(image_path, out_path)


def unit_test2():
    from glob import glob

    model = get_pretrain_nafnet_gopro()
    model = model.to(device)

    fns = glob("%s/*.jpg" % "nafnet_reds_logo_results")
    outpath = "nafnet_reds_gopro_logo_results"
    os.makedirs(outpath, exist_ok=True)
    for f in tqdm(fns):
        ori_img = cv2.imread(f)
        bn = os.path.basename(f)
        ori_img = ori_img.astype(np.float32) / 255.
        ori_img_tensor = img2tensor(ori_img, bgr2rgb=True, float32=True)
        ori_img_tensor = ori_img_tensor.to(device)
        out = model(ori_img_tensor.unsqueeze(0))
        out_img = tensor2img(out.squeeze(0))
        imwrite(out_img, os.path.join(outpath, bn))



if __name__ == '__main__':
    unit_test2()
    pass
