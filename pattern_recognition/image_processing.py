#!/usr/bin/env python3
import numpy as np
import cv2

from skimage.exposure import rescale_intensity


np.seterr(divide='ignore', invalid='ignore')


def convolution(image, kernel):
    """
    Naive implementation of convolution filter.
    """
    (iH, iW) = image.shape  # image height and width
    (kH, kW) = kernel.shape  # kernel height and width
    pad = kH // 2  # tiling window
    # output image with image borders processing
    image = cv2.copyMakeBorder(image, pad, pad, pad, pad,
                               cv2.BORDER_REPLICATE)
    image_padded = np.zeros(shape=(iH + kH + pad, iW + kW + pad))
    image_padded[pad:-pad, pad:-pad] = image
    out = np.zeros(shape=image.shape)  # create output image as original image

    # sliding kernel over the image and compute convolution by element-wise multiplication
    for y in np.arange(pad, iH + pad):
        for x in np.arange(pad, iW + pad):
            for i in range(kH):
                for j in range(kW):
                    out[y, x] += image_padded[y + i, x + j] * kernel[i, j]

    # rescale the output image to be in the range [0, 255]
    out = rescale_intensity(out, in_range=(0, 255))
    out = (out * 255).astype("uint8")
    return out


def sobel_filter(data):
    """
    Implementation of the convolution of a Sobel filter 3x3 with the final goal to obtain edges from data.
    """
    img = cv2.imread(data)
    grey_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    sobelX = np.array(([-1, 0, 1],
                       [-2, 0, 2],
                       [-1, 0, 1]), dtype="int")
    sobelY = np.array(([-1, -2, -1],
                       [0, 0, 0],
                       [1, 2, 1]), dtype="int")
    """
    # shifted kernel matrix to highlight edges in the opposite way
    sobelX = np.array(([+1, 0, -1],
                       [+2, 0, -2],
                       [+1, 0, -1]), dtype="int")
    sobelY = np.array(([+1, +2, +1],
                       [0, 0, 0],
                       [-1, -2, -1]), dtype="int")
    """

    Gx = convolution(grey_img, sobelX)  # horizontal gradient
    Gy = convolution(grey_img, sobelY)  # vertical gradient

    # OpenCV equivalent computations
    Gx_cv2 = cv2.Sobel(grey_img, cv2.CV_8U, 1, 0, ksize=3)
    Gy_cv2 = cv2.Sobel(grey_img, cv2.CV_8U, 0, 1, ksize=3)

    return grey_img, Gx, Gy, Gx_cv2, Gy_cv2
