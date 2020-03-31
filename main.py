#!/usr/bin/env python3
from pattern_recognition.image_processing import sobel_filter
import numpy as np
import cv2
import os

res = 'results/'
os.makedirs(res, exist_ok=True)
data = os.path.abspath("msrc_data/")

scale_factor = 0.8

# iterate over trees and faces images
for filename in os.listdir(data):
    if ((filename.startswith('2_') and filename.endswith('s.bmp')) or
            (filename.startswith('6_') and filename.endswith('s.bmp'))):
        path = os.path.abspath(data + "/" + filename)
        grey_img, Gx, Gy, Gx_cv2, Gy_cv2 = sobel_filter(path)
        magnitude = np.hypot(Gx, Gy)  # gradient magnitude approximation
        exact_gradient = np.hypot(Gx * (1 / 8), Gy * (1 / 8))  # exact gradient magnitude
        direction = cv2.phase(np.float64(Gx), np.float64(Gy), angleInDegrees=True)  # rotation angle
        magnitude_cv2 = np.hypot(Gx_cv2, Gy_cv2)
        direction_cv2 = cv2.phase(np.float64(Gx_cv2), np.float64(Gy_cv2), angleInDegrees=True)

        cv2.imwrite(os.path.join(res, filename + "_grey.jpg"), grey_img)
        cv2.imwrite(os.path.join(res, filename + "_Sobel_X.jpg"), Gx)
        cv2.imwrite(os.path.join(res, filename + "_Sobel_X-scaled.jpg"), Gx * scale_factor)
        cv2.imwrite(os.path.join(res, filename + "_Sobel_X_OpenCV.jpg"), Gx_cv2)
        cv2.imwrite(os.path.join(res, filename + "_Sobel_Y_alt.jpg"), Gy)
        cv2.imwrite(os.path.join(res, filename + "_Sobel_Y-scaled.jpg"), Gy * scale_factor)
        cv2.imwrite(os.path.join(res, filename + "_Sobel_Y_OpenCV.jpg"), Gy_cv2)
        cv2.imwrite(os.path.join(res, filename + "_Sobel_Filter.jpg"), np.float32(magnitude))
        cv2.imwrite(os.path.join(res, filename + "_Sobel_Filter_OpenCV.jpg"), np.float32(magnitude_cv2))
        cv2.imwrite(os.path.join(res, filename + "_Gradient.jpg"), np.float32(exact_gradient))
        cv2.imwrite(os.path.join(res, filename + "_Direction.jpg"), np.float32(direction))
        cv2.imwrite(os.path.join(res, filename + "_Direction_OpenCV.jpg"), np.float32(direction_cv2))
