# -*- coding: utf-8 -*-
"""
Created on Mon Sep 14 04:11:23 2020

@author: Nafis
"""

import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

import os
path = 'C:\\Study\\Filters\\ASL 87k\\asl_alphabet_test\\asl_alphabet_test\\T'
files = os.listdir(path)


for index,file in enumerate(files):
    #print(file)
    img = cv.imread(index,0)
    blurr = cv.GaussianBlur(img,(5,5),0)
    blur = cv.Canny(blurr, 15, 30)

    laplacian = cv.Laplacian(blur,cv.CV_64F)
    sobelx = cv.Sobel(blur,cv.CV_64F,1,0,ksize=5)
    sobely = cv.Sobel(blur,cv.CV_64F,0,1,ksize=5)
    plt.subplot(2,2,1),plt.imshow(blur,cmap = 'gray')
    plt.title('Original'), plt.xticks([]), plt.yticks([])
    plt.subplot(2,2,2),plt.imshow(laplacian,cmap = 'gray')
    plt.title('Laplacian'), plt.xticks([]), plt.yticks([])
    plt.subplot(2,2,3),plt.imshow(sobelx,cmap = 'gray')
    plt.title('Sobel X'), plt.xticks([]), plt.yticks([])
    plt.subplot(2,2,4),plt.imshow(sobely,cmap = 'gray')
    plt.title('Sobel Y'), plt.xticks([]), plt.yticks([])
    plt.show()



