import os
import cv2

from bayesian_matting import BayesianMatting
Demo = BayesianMatting()
img = cv2.imread('gandalf.png')
trimap = cv2.imread('gandalfTrimap.png', 0)
Demo.bayesian_matte(img, trimap)
