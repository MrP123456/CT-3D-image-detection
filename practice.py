import sys

import numpy as np
import matplotlib.pyplot as plt
import cv2


if __name__=='__main__':
    img=np.array([[0,1,0,0],[0,1,1,0],[0,0,1,0],[0,0,0,0]])
    img[img>0]=255
    img=img.astype('uint8')
    img=cv2.resize(img,[256,256])
    cv2.imshow('1',img)
    cv2.waitKey()

    cnts,_=cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    img=cv2.cvtColor(img,cv2.COLOR_GRAY2RGB)
    cv2.drawContours(img,cnts,-1,(0,0,255),2)
    points=np.squeeze(cnts[0]).astype('float32')
    print(points.shape)
    mean, eigenvectors, eigenvalues = cv2.PCACompute2(points,  np.array([]))
    center = mean[0, :].astype(int)
    e1xy = eigenvectors[0, :] * eigenvalues[0, 0]  # 第一主方向轴
    p1 = (center + 0.01 * e1xy).astype(np.int)  # P1:[149 403]
    theta = np.arctan2(eigenvectors[0, 1], eigenvectors[0, 0]) * 180 / np.pi
    cv2.arrowedLine(img, center, p1, (255, 0, 0), thickness=3, tipLength=0.1)
    print(theta)

    cv2.imshow('1',img)
    cv2.waitKey()
