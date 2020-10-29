import os
import cv2
import numpy as np

from poisson_edit import PoissonImageEdit

source_img = cv2.imread('input/2/source.jpg')
target_img = cv2.imread('input/2/target.jpg')
source_img = source_img.astype(np.float)
target_img = target_img.astype(np.float)
mask = cv2.imread('input/2/mask.jpg', 0)
mask = np.atleast_3d(mask).astype(np.float) / 255.
mask[mask != 1] = 0

demo = PoissonImageEdit()
channels = source_img.shape[-1]
result_stack = [demo.process(source_img[:,:,i], target_img[:,:,i], mask) for i in range(channels)]
result = cv2.merge(result_stack)
#result = demo.process(source_img, target_img, mask)
#result = result.astype('uint8')
cv2.imwrite('result.png', result)
