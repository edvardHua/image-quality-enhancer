# -*- coding: utf-8 -*-
# @Time : 2023/5/5 16:20
# @Author : zihua.zeng
# @File : bm3d.py

import cv2
import numpy as np
import math


def process_bar(percent, start_str='', end_str='', total_length=0):
    bar = ''.join(["\033[31m%s\033[0m" % '   '] * int(percent * total_length)) + ''
    bar = '\r' + start_str + bar.ljust(total_length) + ' {:0>4.1f}%|'.format(percent * 100) + end_str
    print(bar, end='', flush=True)


image = cv2.imread('assets/a0016.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

size = image.shape  # h w c

# 生成噪声
image = np.array(image / 255, dtype=np.float32)
noise = np.random.normal(0, 0.001 ** 0.5, size)
imnoise = image + 1 * noise

# cv2.imshow('result',imnoise)
# cv2.waitKey()

sigma = 1
searchsize = 16  # 图像搜索区域半径
blocksize = 4  # 图像块尺寸
blockstep = 2  # 搜索步长
blockmaxnum = 8  # 相似块最大数量
searchblocksize = searchsize / blocksize  # 半径内块个数
kai = np.kaiser(blocksize, 5)
kai = np.dot(kai[:, None], kai[None, :])  # 二维kaiser

diffT = 100  # 允许纳入数组的最大diff
coefT = 0.0005

diffArray = np.zeros(blockmaxnum)  # 相似块
similarBoxArray = np.zeros((blocksize, blocksize, blockmaxnum))
# jinbuqu csdn newland


# 让图像尺寸符合倍数
newh = math.floor((size[0] - blocksize) / blockstep) * blockstep - searchsize
neww = math.floor((size[1] - blocksize) / blockstep) * blockstep - searchsize

newh = math.floor((size[0] - newh - blocksize) / blockstep) * blockstep + newh
neww = math.floor((size[1] - neww - blocksize) / blockstep) * blockstep + neww

imnoise = imnoise[0:newh, 0:neww]

# 初始化分子分母
imnum = np.zeros(imnoise.shape)
imden = np.zeros(imnoise.shape)

# 将左上角作为块的坐标.每个块独立做一次去噪，只要关注一块的计算就行
for y in range(0, newh - blocksize, blockstep):  # 检查能不能完整走到底,范围对不对
    systart = max(0, y - searchsize)
    syend = min(newh - blocksize, y + searchsize - 1)
    process_bar(y / (newh - blocksize), start_str='进度', end_str="100", total_length=15)
    for x in range(0, neww - blocksize, blockstep):

        sxstart = max(0, x - searchsize)
        sxend = min(neww - blocksize, x + searchsize - 1)

        # 排序矩阵初始化
        similarBoxArray[:, :, 0] = imnoise[y: y + blocksize, x: x + blocksize]
        hasboxnum = 1
        diffArray[0] = 0
        # 不算自己
        # 不算自己
        for sy in range(systart, syend, blockstep):
            for sx in range(sxstart, sxend, blockstep):
                if sy == y and sx == x:
                    continue

                diff = np.sum(np.abs(
                    imnoise[y: y + blocksize, x: x + blocksize] - imnoise[sy: sy + blocksize, sx: sx + blocksize]))

                if diff > diffT:
                    continue
                # 塞入

                changeid = 0
                if hasboxnum < blockmaxnum - 1:
                    changeid = hasboxnum
                    hasboxnum = hasboxnum + 1
                else:
                    # 排序
                    for difid in range(1, blockmaxnum - 1):

                        if diff < diffArray[difid]:
                            changeid = difid

                if changeid != 0:
                    similarBoxArray[:, :, changeid] = imnoise[sy: sy + blocksize, sx: sx + blocksize]
                    diffArray[changeid] = diff

        # 开始做dct
        for difid in range(1, hasboxnum):
            similarBoxArray[:, :, difid] = cv2.dct(similarBoxArray[:, :, difid])

        # 开始做1d,阈值操作，计算非零个数
        notzeronum = 0
        for y1d in range(0, blocksize):
            for x1d in range(0, blocksize):
                temp3ddct = cv2.dct(similarBoxArray[y1d, x1d, :])
                zeroidx = np.abs(temp3ddct) < coefT

                temp3ddct[zeroidx] = 0
                notzeronum = notzeronum + temp3ddct[temp3ddct != 0].size
                similarBoxArray[y1d, x1d, :] = cv2.idct(temp3ddct)[:, 0]

        # jinbuqu csdn newland

        if notzeronum < 1:
            notzeronum = 1

        # 最终恢复图像所用到的分子分母
        for difid in range(1, hasboxnum):
            # 求权重：kaiser * w
            weight = kai / ((sigma ** 2) * notzeronum)
            imidct = cv2.idct(similarBoxArray[:, :, difid])

            # 分子，分母
            imnum[y: y + blocksize, x: x + blocksize] += imidct * weight
            imden[y: y + blocksize, x: x + blocksize] += weight

        x = x + blocksize

    y = y + blocksize

# 完成去噪
imout = imnum / imden

cv2.imshow('in', imnoise)
cv2.waitKey(0)
cv2.imshow('out', imout)
cv2.waitKey(0)
