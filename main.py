from visualization_tools import draw_origin_on_image
import numpy as np
import cv2

im = cv2.imread("/home/yaroslav/Desktop/door_handle_detection/dataset/rgb_30c71e67e1d2e46c268fba41687e1d3b.jpg")
im = cv2.imread("/home/yaroslav/Desktop/door_handle_detection/dataset/rgb_8109ae0b1b22faa5dad619176e7433e7.jpg")
pose = np.load("/home/yaroslav/Desktop/door_handle_detection/dataset/pose_30c71e67e1d2e46c268fba41687e1d3b.npy")
pose = np.load("/home/yaroslav/Desktop/door_handle_detection/dataset/pose_8109ae0b1b22faa5dad619176e7433e7.npy")
im1 = draw_origin_on_image(im, pose)
im1 = cv2.resize(im1, (im1.shape[1] // 2, im1.shape[0] // 2))
cv2.imshow("WIN", im1)
cv2.waitKey()
