import cv2
import numpy as np
import glob

# Chessboard dimensions
chessboard_size = (12, 12)  # Number of inner corners per a chessboard row and column
square_size = 0.0373575 / 2  # Set this to your actual square size

# Arrays to store object points and image points
objpoints = []  # 3D points in real world space
imgpoints = []  # 2D points in image plane

# Prepare object points (0,0,0), (1,0,0), (2,0,0) ..., (6,5,0)
objp = np.zeros((np.prod(chessboard_size), 3), np.float32)
objp[:, :2] = np.indices(chessboard_size).T.reshape(-1, 2)
objp *= square_size

cv2.ocl.setUseOpenCL(False)
for i in range(1976):
    if i % 20 != 0:
        continue
    print(f"i = {i}")
    path = f"/Users/hladunyaroslav/Desktop/door_handle_labeling/rgb1/{i}.jpg"
    img = cv2.imread(path)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Find the chess board corners
    ret, corners = cv2.findChessboardCorners(gray, chessboard_size)

    # If found, add object points, image points
    if ret:
        objpoints.append(objp)
        imgpoints.append(corners)

        # Draw and display the corners
        cv2.drawChessboardCorners(img, chessboard_size, corners, ret)
        cv2.imshow('Chessboard Corners', img)
        cv2.waitKey(500)  # Pause to show images

# Camera calibration
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

# The matrix 'mtx' is your intrinsic camera matrix
print("Camera Matrix:\n", mtx)

# 'dist' are the distortion parameters
print("Distortion Coefficients:\n", dist)

# Camera Matrix:
#  [[ 1387.56  0        708.89]
#  [  0        1387.96  956.88]
#  [  0        0        1]]

# Distortion Coefficients:
#  [[ 2.03226134e-01 -7.04873546e-01 -3.51671357e-05 -8.49750941e-04
#    6.28718638e-01]]

# Camera Matrix:
#  [[1361.94 0.00000000e+00 707]
#  [0.00000000e+00 1.36146549e+03 9.60978490e+02]
#  [0.00000000e+00 0.00000000e+00 1.00000000e+00]]

# Distortion Coefficients:
#  [[ 1.84712073e-01 -4.72102673e-01 -6.50624249e-04  7.78799393e-05
#   -3.05262249e-01]]