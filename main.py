from visualization_tools import draw_origin_on_image
import numpy as np
import cv2

im = cv2.imread("/home/yaroslav/Desktop/door_handle_detection/dataset/rgb_8109ae0b1b22faa5dad619176e7433e7.jpg")
pose = np.load("/home/yaroslav/Desktop/door_handle_detection/dataset/pose_8109ae0b1b22faa5dad619176e7433e7.npy")
im1 = draw_origin_on_image(im, pose)
cv2.imshow("WIN", im1)
cv2.waitKey()