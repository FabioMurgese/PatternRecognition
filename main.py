#!/usr/bin/env python3
from pattern_recognition.image_processing import sobel_filter, normalized_cut, sift_descriptor
from pattern_recognition.signal_processing import autoregressive_analysis, correlation_temperatures

import numpy as np
import cv2 as cv
import os
from matplotlib import pyplot as plt
from skimage.io import imshow

energy_data = os.path.abspath('datasets/energy_data/energydata_complete.csv')
autoregressive_analysis(energy_data)
correlation_temperatures(energy_data)

kp, sift_desc = sift_descriptor('datasets/msrc_data/1_1_s.bmp')
imshow(sift_desc)
plt.show()

"""fig = normalized_cut('datasets/msrc_data/1_1_s.bmp')
imshow("Normalized cut image", fig)
plt.show()"""

res = 'results/'
os.makedirs(res, exist_ok=True)
img_data = os.path.abspath("datasets/msrc_data/")
scale_factor = 0.8

# iterate over trees and faces images
for filename in os.listdir(img_data):
    if ((filename.startswith('2_') and filename.endswith('s.bmp')) or
            (filename.startswith('6_') and filename.endswith('s.bmp'))):
        path = os.path.abspath(img_data + "/" + filename)
        grey_img, Gx, Gy, Gx_cv2, Gy_cv2 = sobel_filter(path)
        magnitude = np.hypot(Gx, Gy)  # gradient magnitude approximation
        exact_gradient = np.hypot(Gx * (1 / 8), Gy * (1 / 8))  # exact gradient magnitude
        direction = cv.phase(np.float64(Gx), np.float64(Gy), angleInDegrees=True)  # rotation angle
        magnitude_cv2 = np.hypot(Gx_cv2, Gy_cv2)
        direction_cv2 = cv.phase(np.float64(Gx_cv2), np.float64(Gy_cv2), angleInDegrees=True)

        cv.imwrite(os.path.join(res, filename + "_grey.jpg"), grey_img)
        cv.imwrite(os.path.join(res, filename + "_Sobel_X.jpg"), Gx)
        cv.imwrite(os.path.join(res, filename + "_Sobel_X-scaled.jpg"), Gx * scale_factor)
        cv.imwrite(os.path.join(res, filename + "_Sobel_X_OpenCV.jpg"), Gx_cv2)
        cv.imwrite(os.path.join(res, filename + "_Sobel_Y.jpg"), Gy)
        cv.imwrite(os.path.join(res, filename + "_Sobel_Y-scaled.jpg"), Gy * scale_factor)
        cv.imwrite(os.path.join(res, filename + "_Sobel_Y_OpenCV.jpg"), Gy_cv2)
        cv.imwrite(os.path.join(res, filename + "_Sobel_Filter.jpg"), np.float32(magnitude))
        cv.imwrite(os.path.join(res, filename + "_Sobel_Filter_OpenCV.jpg"), np.float32(magnitude_cv2))
        cv.imwrite(os.path.join(res, filename + "_Gradient.jpg"), np.float32(exact_gradient))
        cv.imwrite(os.path.join(res, filename + "_Direction.jpg"), np.float32(direction))
        cv.imwrite(os.path.join(res, filename + "_Direction_OpenCV.jpg"), np.float32(direction_cv2))
