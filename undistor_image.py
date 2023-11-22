import cv2
import numpy as np

# Load your image
image = cv2.imread('/Users/hladunyaroslav/Desktop/door_handle_labeling/rgb1/648.jpg')

# Camera matrix and distortion coefficients
camera_matrix = np.array([[1.36194252e+03, 0, 7.07090483e+02], [0, 1.36146549e+03, 9.60978490e+02], [0, 0, 1]])
dist_coeffs = np.array([2.03226134e-01, -7.04873546e-01, -3.51671357e-05, -8.49750941e-04, 6.28718638e-01])

# Undistort the image
undistorted_image = cv2.undistort(image, camera_matrix, dist_coeffs, None, camera_matrix)
cv2.imshow("win", undistorted_image)
cv2.imshow("win1", image)
cv2.waitKey()