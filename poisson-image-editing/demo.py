import os
import cv2

from poisson_edit import PoissonImageEdit

source_img = cv2.imread('input/2/source.jpg')
target_img = cv2.imread('input/2/target.jpg')
mask = cv2.imread('input/2/mask.jpg', 0)

demo = PoissonImageEdit()
result = demo.process(source_img, target_img, mask)
result = result.astype('uint8')
cv2.imwrite('result.png', result)
