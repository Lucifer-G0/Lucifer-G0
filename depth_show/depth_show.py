import numpy as np 
import argparse
import cv2
from matplotlib import pyplot as plt


def depth_to_uint8(dI):
    """Simply handles converting a depth channel to 0-255"""
    return (255 * (dI / float(dI.max()))).astype(np.uint8)

ap = argparse.ArgumentParser()
ap.add_argument("-i","--image",required = True,help = "Path to the image")
args = vars(ap.parse_args())

image = cv2.imread(args["image"],-1).astype(np.float)
image = depth_to_uint8(image)
cv2.imshow("Original",image)


hist = cv2.calcHist([image],[0],None,[256],[0,256])

plt.figure()
plt.title("Grayscale Histogram")
plt.xlabel("Bins")
plt.ylabel("# of Pixels")
plt.plot(hist)
plt.xlim([0,256])
plt.show()
cv2.waitKey(0)


# plt.imshow(image,interpolation='nearest',cmap='bone',origin='lower')
# #根据像素绘制图片 origin表示渐变程度
# plt.colorbar()
# #显示像素与数据对比
# plt.xticks(())
# plt.yticks(())
# #不显示坐标轴刻度
# plt.show()
# #显示图片

