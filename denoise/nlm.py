# -*- coding: utf-8 -*-
# @Time : 2023/5/5 16:19
# @Author : zihua.zeng
# @File : nlm.py


import cv2
import numpy as np


def psnr(A, B):
    return 10 * np.log(255 * 255.0 / (((A.astype(np.float) - B) ** 2).mean())) / np.log(10)


def double2uint8(I, ratio=1.0):
    return np.clip(np.round(I * ratio), 0, 255).astype(np.uint8)


def make_kernel(f):
    """
    生成高斯核，越靠近中心位置的像素，权重越高
    """
    kernel = np.zeros((2 * f + 1, 2 * f + 1))
    for d in range(1, f + 1):
        kernel[f - d:f + d + 1, f - d:f + d + 1] += (1.0 / ((2 * d + 1) ** 2))
    return kernel / kernel.sum()


def NLmeansfilter(I, h_=10, templateWindowSize=5, searchWindowSize=11):
    f = templateWindowSize // 2
    t = searchWindowSize // 2
    height, width = I.shape[:2]
    padLength = t + f
    I2 = np.pad(I, padLength, 'symmetric')
    kernel = make_kernel(f)
    h = (h_ ** 2)
    I_ = I2[padLength - f:padLength + f + height, padLength - f:padLength + f + width]

    average = np.zeros(I.shape)
    sweight = np.zeros(I.shape)
    wmax = np.zeros(I.shape)
    for i in range(-t, t + 1):
        for j in range(-t, t + 1):
            if i == 0 and j == 0:
                continue
            I2_ = I2[padLength + i - f:padLength + i + f + height, padLength + j - f:padLength + j + f + width]
            w = np.exp(-cv2.filter2D((I2_ - I_) ** 2, -1, kernel) / h)[f:f + height, f:f + width]
            sweight += w
            wmax = np.maximum(wmax, w)
            average += (w * I2_[f:f + height, f:f + width])
    # 原图像需要乘于最大权重参与计算
    average += (wmax * I)
    # sweight 为 weight 之和，用于计算 weight 的归一化
    sweight += wmax
    # 除于 sweight 获得均值，也就是降噪后的图
    return average / sweight


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    I = cv2.imread('assets/a0016.jpg', 0)
    sigma = 20.0
    I1 = double2uint8(I + np.random.randn(*I.shape) * sigma)

    plt.imshow(I1, cmap="gray")
    plt.show()

    print('噪声图像PSNR', psnr(I, I1))
    R1 = cv2.medianBlur(I1, 5)
    print('中值滤波PSNR', psnr(I, R1))
    R2 = cv2.fastNlMeansDenoising(I1, None, sigma, 5, 11)
    print('opencv的NLM算法', psnr(I, R2))

    plt.imshow(R2, cmap="gray")
    plt.show()

    R3 = double2uint8(NLmeansfilter(I1.astype(np.float), sigma, 5, 11))
    print('NLM PSNR', psnr(I, R3))
    plt.imshow(R3, cmap="gray")
    plt.show()
