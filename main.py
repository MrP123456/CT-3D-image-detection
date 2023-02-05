import sys

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
    '''img = data[1100, :, :]

    _, img = cv2.threshold(img, 150, 255, cv2.THRESH_BINARY)
    hx, hy = np.sum(img, 0), np.sum(img, 1)
    rx, ry = np.arange(len(hx)), np.arange(len(hy))
    dx, dy = int(np.round(np.sum(hx * rx) / np.sum(hx))), int(np.round(np.sum(hy * ry) / np.sum(hy)))
    # print(dx, dy) 317 286
    print(dx, dy)'''

    img = data[1100, :, :]
    _, img = cv2.threshold(img, 150, 255, cv2.THRESH_BINARY)
    cnts, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cnt = cnts[0]
    for i in range(1, len(cnts)):
        cnt = np.concatenate([cnt, cnts[i]], 0)
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    points = np.squeeze(cnt).astype('float32')
    mean, eigenvectors, eigenvalues = cv2.PCACompute2(points, np.array([]))
    center = mean[0, :].astype(int)
    e1xy = eigenvectors[0, :] * eigenvalues[0, 0]  # 第一主方向轴
    p1 = (center + 0.01 * e1xy).astype('int32')  # P1:[149 403]
    thet = np.arctan2(eigenvectors[0, 1], eigenvectors[0, 0]) * 180 / np.pi
    cv2.arrowedLine(img, center, p1, (255, 0, 0), thickness=20, tipLength=1)
    img=cv2.resize(img,[256,256])
    cv2.imshow('角度',img)
    cv2.waitKey()

    # 旋转图像
    # thet = -np.arctan((mx - dx) / (my - dy)) / 2 / np.pi * 360
    M = cv2.getRotationMatrix2D((mx, my), -thet, 1)

    out = np.zeros([1536, 1536]).astype('uint8')
    for i, img in enumerate(data):
        img = cv2.warpAffine(img, M, [1536, 1536])
        out[i] = img[:, my]

    # out=data[:,1536//2,:]
    cv2.imshow('out',out)
    cv2.waitKey()

    cv2.imwrite(img_path, out)


def calcu_dis(path):
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    print(img.shape)
    h, w = img.shape
    img = img[h // 4:-h // 4]
    _, img = cv2.threshold(img, 150, 255, cv2.THRESH_BINARY)
    img = 255 - img
    img = img[:, :img.shape[1] // 2]


    # print(img.shape) [768,1536]
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(img)
    num = 0
    out = np.zeros_like(labels).astype('uint8')
    idxs = []
    for i in range(1, num_labels):
        if 10 <= stats[i, -1] < 100:
            out[labels == i] = 255
            num += 1
            idxs.append(i)
    print(num)
    print(num_labels)

    cnts, _ = cv2.findContours(out, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    num = len(cnts)
    sort_mat = np.zeros(num)
    for i in range(num):
        Box2D = cv2.minAreaRect(cnts[i])
        sort_mat[i] = 1 * Box2D[0][0] + 3 * Box2D[0][1]
    idx = np.argsort(sort_mat)

    assert num % 3 == 0
    num_x, num_y = num // 3, 3
    idx = idx.reshape([num_x, num_y])
    dis_mat = np.zeros([num_x, 4])

    for i in range(num_x):
        for j in range(num_y):
            cnt = cnts[idx[i, j]]
            Box2D = cv2.minAreaRect(cnt)
            print(int(Box2D[0][0]), int(Box2D[0][1]), end='  ')
        print()

    out = cv2.cvtColor(out, cv2.COLOR_GRAY2RGB)

    for i in range(num_x):
        id = idx[i, 0]
        cnt = cnts[id]
        left_point = tuple(cnt[cnt[:, :, 0].argmin()][0])
        y, x = left_point
        ty = y - 1
        while True:
            p = False
            px, py = 0, 0
            for k in range(-5, 5):
                xx, yy = x + k, ty
                if xx < 0 or xx >= len(img):
                    continue
                if img[xx, yy] != 0:
                    p = True
                    px, py = xx, yy
                    break
            if p:
                dis = np.sqrt((x - px) ** 2 + (y - py) ** 2)
                break
            ty -= 1
        dis_mat[i, 0] = dis

        right_point = tuple(cnt[cnt[:, :, 0].argmax()][0])
        y, x = right_point
        ty = y + 1
        while True:
            p = False
            px, py = 0, 0
            for k in range(-5, 5):
                xx, yy = x + k, ty
                if xx < 0 or xx >= len(img):
                    continue
                if img[xx, yy] != 0:
                    p = True
                    px, py = xx, yy
                    break
            if p:
                dis = np.sqrt((x - px) ** 2 + (y - py) ** 2)
                break
            ty += 1
        dis_mat[i, -1] = dis

        for j in range(0, 2):
            id1, id2 = idx[i, j], idx[i, j + 1]
            cnt1, cnt2 = cnts[id1], cnts[id2]
            dis = min_dis(cnt1, cnt2)
            '''
            if dis>100:
                cv2.drawContours(out,[cnt1,cnt2],-1,(0,0,255),2)
                cv2.imshow(str(i),out)
                cv2.waitKey()'''

            ''' left_point = tuple(cnt1[cnt1[:, :, 0].argmax()][0])
            right_point = tuple(cnt2[cnt2[:, :, 0].argmin()][0])
            (x1, y1), (x2, y2) = left_point, right_point
            dis2 = np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

            dis_mat[i, j + 1] = dis
            cv2.circle(out, (int(x1), int(y1)), 1, (0, 0, 255), -1)
            cv2.circle(out, (int(x2), int(y2)), 1, (0, 0, 255), -1)'''
            dis_mat[i, j + 1] = dis

    print(dis_mat)
    cv2.imshow('out', img)
    cv2.waitKey()

    '''num_labels = num
    stats = stats[idxs]
    centroids = centroids[idxs]



    temp = np.array([1, 5, 0, 0, 0]).reshape([1, -1])
    q = np.sum(stats * temp, 1)
    sort_idx = np.argsort(q)
    stats = stats[sort_idx]
    centroids = centroids[sort_idx]
    assert num_labels % 3 == 0
    h, w = num_labels // 3, 3
    pos = stats[:, :2]
    pos = pos[:, ::-1]
    pos=pos.reshape([h,w,2])
    # print(pos) [38,3,2]
    dis_mat=np.zeros([len(pos),5])
    for i in range(len(pos)):
        idx=i*3
        x,y=stats[idx,1],stats[idx,0]
        id=labels[x,y]
        left_x,left_y=stats'''


def min_dis(cnt1, cnt2):
    min_dis = 99999
    for i in range(len(cnt1)):
        point = cnt1[i][0]
        dis = cv2.pointPolygonTest(cnt2, (int(point[0]), int(point[1])), True)
        min_dis = min(min_dis, np.abs(dis))
    return min_dis


if __name__ == '__main__':
    raw_path = 'dataset/0/FdkRecon-ushort-1536x1536x1536.raw'
    array_path = 'dataset/data.npy'
    img_path = 'dataset/img.png'

    # 将raw转array
    # preprocess(raw_path,array_path)
    # 从三维数组中得到想要的切片图
    # get_img(array_path, img_path)
    calcu_dis(img_path)
