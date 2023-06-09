# -*- coding: utf-8 -*-
# @Time : 2023/6/9 16:25
# @Author : zihua.zeng
# @File : infer_restoration.py

import cv2
import numpy as np
from restoration.nafnet_reds import get_pretrain_nafnet_reds
from utils.img_util import img2tensor, tensor2img, imwrite


def test_jpg_compress_by_nafnet(input_image, output_image):
    model = get_pretrain_nafnet_reds()
    ori_img = cv2.imread(input_image)
    ori_img = ori_img.astype(np.float32) / 255.
    ori_img_tensor = img2tensor(ori_img, bgr2rgb=True, float32=True)
    out = model(ori_img_tensor.unsqueeze(0))
    out_img = tensor2img(out.squeeze(0))
    imwrite(out_img, output_image)


if __name__ == '__main__':
    image_path = "/Users/zihua.zeng/Pictures/7a6a8d608f0579e71a53f9aadf1c3b99.jpg"
    out_path = "/Users/zihua.zeng/Pictures/out_7a6a8d608f0579e71a53f9aadf1c3b99.jpg"
    test_jpg_compress_by_nafnet(image_path, out_path)
    pass
