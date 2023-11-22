BASE_PATH = "EXR_RGBD"

import json
import cv2
import numpy as np
import OpenEXR
import Imath
import matplotlib.pyplot as plt
import math
from typing import Tuple


def ijd_to_xyz(i, j, d, cam_info):
    # Extract the focal lengths and principal point coordinates from cam_info
    fx, fy = cam_info[0, 0], cam_info[1, 1]
    px, py = cam_info[0, 2], cam_info[1, 2]

    # Convert from image coordinates (i, j, d) to 3D world coordinates (X, Y, Z)
    X = (i - px) * d / fx
    Y = (j - py) * d / fy
    Z = d

    return np.array([X, Y, Z])


# metadata_path = f"{BASE_PATH}/metadata_x.json"
metadata_path = f"metadata.json"

with open(metadata_path, 'r') as file:
    metadata = json.load(file)

print(metadata.keys())
print(metadata['initPose'])
print(metadata['K'])
camera_matrix = np.array(metadata['K']).reshape(3, 3).T
dist_coeffs = np.array([2.03226134e-01, -7.04873546e-01, -3.51671357e-05, -8.49750941e-04, 6.28718638e-01])
handle_pose = ijd_to_xyz(595, 718, 0.37, camera_matrix)

images = []
# for i in range(len(metadata["frameTimestamps"])):
# print(d[128, 96])
# exit(0)

# Last three values.
# 5 of camera look top
# 4 look right
poses = np.array(metadata['poses'])
for i in range(7):
    plt.plot(poses[:, i], label=f"{i}")
plt.legend()
plt.show()
# poses[:, 4] = -poses[:, 4]
# poses[:, 5] = -poses[:, 5]
for i in range(1000):
    rgb_image_path = f"{BASE_PATH}/rgb/{i}.jpg"
    image = cv2.imread(rgb_image_path)
    depth_image = read_exr(f"{BASE_PATH}/depth/{i}.exr")
    # plt.imshow(cv2.resize(depth_image, (images[-1].shape[1], images[-1].shape[0])))
    # plt.show()

    pose = handle_pose + poses[0][-3:] - poses[i][-3:]
    # pose = np.array([0, 0, 0]) - metadata['poses'][i][:3]
    image_with_axes = draw_origin_on_image(
        image,
        camera_matrix,
        pose,
        None,
        0.5
    )
    #
    # # Display the image
    cv2.imshow('Image with 3D Axes', image_with_axes)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
