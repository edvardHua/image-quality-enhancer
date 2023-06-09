# -*- coding: utf-8 -*-
# @Time : 2023/5/15 21:36
# @Author : zihua.zeng
# @File : test.py

import cv2
import numpy as np
import matplotlib.pyplot as plt

# Set up a figure twice as tall as it is wide
fig = plt.figure()
fig.suptitle('DCT Visualization', fontsize=16)

# Second subplot
ax = fig.add_subplot(1, 1, 1, projection='3d')

image = cv2.imread("assets/a0016.jpg", 0)
# resize 到 100 方便后面可视化
image = cv2.resize(image, (100, 100))
# 加入噪音
# size = image.shape
# noise = np.random.normal(0, 0.001 ** 0.5, size)
# image = image + 1 * noise

img_dct = np.float32(image)
dct = cv2.dct(img_dct)
# 方便可视化，这里除于100，并把大于10的都置零
dct_out = abs(dct) / 100
dct_out[dct_out > 8] = 0

h, w = dct_out.shape

X = np.arange(0, w, 1)
Y = np.arange(0, h, 1)
X, Y = np.meshgrid(X, Y)

surf = ax.plot_surface(X, Y, dct_out, rstride=1, cstride=1,
                       linewidth=0, antialiased=False)
ax.set_zlim(-4, 8)

plt.show()
