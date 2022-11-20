
import os
import numpy as np
from PIL import Image
import io, base64
import cv2,copy


def changedH(bgr_img, shift):
    hsvimage = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2HSV_FULL) # BGR->HSV
    hi = hsvimage[:,:,0].astype(np.int32)
    if shift < 0:
        nhi = hi.flatten()
        for px in nhi:
            if px < 0:
                px = 255 - px
        nhi = nhi.reshape(hsvimage.shape[:2])
        hi = nhi.astype(np.uint8)
    chimg = (hi + shift) % 255
    hsvimage[:,:,0] = chimg
    hsv8 = hsvimage.astype(np.uint8)
    return cv2.cvtColor(hsv8,cv2.COLOR_HSV2BGR_FULL) # HSV->BGR

# HSV S(彩度),V(明度)の変更
def changedSV(bgr_img, alpha, beta, color_idx):
    hsvimage = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2HSV_FULL) # BGR->HSV
    hsvf = hsvimage.astype(np.float32)
    hsvf[:,:,color_idx] = np.clip(hsvf[:,:,1] * alpha+beta, 0, 255)
    hsv8 = hsvf.astype(np.uint8)
    return cv2.cvtColor(hsv8,cv2.COLOR_HSV2BGR_FULL)

# HSV S(彩度)の変更
def changedS(bgr_img, alpha, beta):
    return changedSV(bgr_img, alpha, beta, 1)

# HSV V(明度)の変更
def changedV(bgr_img, alpha, beta):
    return changedSV(bgr_img, alpha, beta, 2)