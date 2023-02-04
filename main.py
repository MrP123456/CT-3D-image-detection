import numpy as np
import matplotlib.pyplot as plt
import cv2


def preprocess(path, to_path):
    img = np.fromfile(path, dtype='uint8')
    img = img.reshape([1536, 1536, 1536, 2])
    img = img[:, :, :, 1]
    np.save(to_path, img)


def get_img(path, img_path):
    data = np.load(path)

    # defect_img = data[100, :, :]

    # 检测出圆心坐标(x,y)
    circle_img = data[500, :, :]
    circle_img[circle_img > 150] = 255
    circle_img[circle_img <= 150] = 0
    circle_img = cv2.resize(circle_img, [1536, 1536])
    circles = cv2.HoughCircles(circle_img, cv2.HOUGH_GRADIENT, dp=1, minDist=1000, param1=100, param2=30)
    mx, my, r = int(np.round(circles[0, 0, 0])), int(np.round(circles[0, 0, 1])), circles[0, 0, 2]
    # print(mx, my, r) 766.0 764.0 658.2

    # 检测出缺陷点坐标(x,y)
    img = data[100, :, :]
    _, img = cv2.threshold(img, 150, 255, cv2.THRESH_BINARY)
    hx, hy = np.sum(img, 0), np.sum(img, 1)
    rx, ry = np.arange(len(hx)), np.arange(len(hy))
    dx, dy = int(np.round(np.sum(hx * rx) / np.sum(hx))), int(np.round(np.sum(hy * ry) / np.sum(hy)))
    # print(dx, dy) 317 286
    '''img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    cv2.circle(img, [dx, dy], 10, (0, 0, 255))
    cv2.imshow('1', img)
    cv2.waitKey()'''

    # 旋转图像
    thet = -np.arctan((mx - dx) / (my - dy)) / 2 / np.pi * 360
    M = cv2.getRotationMatrix2D((mx, my), thet, 1)

    out = np.zeros([1536, 1536]).astype('uint8')
    for i, img in enumerate(data):
        img = cv2.warpAffine(img, M, [1536, 1536])
        out[i] = img[:, my]

    # out=data[:,1536//2,:]
    cv2.imwrite(img_path, out)
    cv2.imshow('out', out)
    cv2.waitKey()


if __name__ == '__main__':
    raw_path = 'dataset/0/FdkRecon-ushort-1536x1536x1536.raw'
    array_path = 'dataset/data.npy'
    img_path = 'dataset/img.png'

    # 将raw转array
    # preprocess(raw_path,array_path)
    # 从三维数组中得到想要的切片图
    get_img(array_path, img_path)
