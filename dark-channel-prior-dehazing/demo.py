import os
import cv2

from dehaze import HazeRemovalUsingDarkChannelPrior

Demo = HazeRemovalUsingDarkChannelPrior()
img = cv2.imread('15.png')
dark, radiance, A, t = Demo.dehaze(img)
cv2.imwrite('dark.png', dark)
cv2.imwrite('radiance.png', radiance)
