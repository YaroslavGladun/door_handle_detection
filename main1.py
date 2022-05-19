import numpy
import pyrealsense2 as rs
import numpy as np
import cv2
from door_handle_detection_utils import RANSACProcessor
from door_handle_detection_utils import DepthImageRANSACProcessor

indexes1 = np.array([i for i in range(480)]).reshape(-1, 1)
indexes1 = np.repeat(indexes1, 640, axis=1)
indexes1 = indexes1.reshape(*indexes1.shape, 1)
indexes2 = np.array([i for i in range(640)]).reshape(-1, 1)
indexes2 = np.repeat(indexes2, 480, axis=1).T
indexes2 = indexes2.reshape(*indexes2.shape, 1)
indexes = np.concatenate((indexes1, indexes2, np.ones((480, 640, 1))), axis=2)

pipeline = rs.pipeline()
config = rs.config()
pipeline_wrapper = rs.pipeline_wrapper(pipeline)
pipeline_profile = config.resolve(pipeline_wrapper)
device = pipeline_profile.get_device()
device_product_line = str(device.get_info(rs.camera_info.product_line))

found_rgb = False
for s in device.sensors:
    if s.get_info(rs.camera_info.name) == 'RGB Camera':
        found_rgb = True
        break
if not found_rgb:
    print("The demo requires Depth camera with Color sensor")
    exit(0)

config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

if device_product_line == 'L500':
    config.enable_stream(rs.stream.color, 960, 540, rs.format.bgr8, 30)
else:
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

profile = pipeline.start(config)

depth_sensor = profile.get_device().first_depth_sensor()
depth_scale = depth_sensor.get_depth_scale()
print("Depth Scale is: ", depth_scale)

clipping_distance_in_meters = 1
clipping_distance = clipping_distance_in_meters / depth_scale

align_to = rs.stream.color
align = rs.align(align_to)
pc = rs.pointcloud()

ransac_processor = RANSACProcessor(5000, 2500)
depth_image_ransac_processor = DepthImageRANSACProcessor(1000, 500)

depth_image = 0
try:
    while True:
        frames = pipeline.wait_for_frames()
        aligned_frames = align.process(frames)
        aligned_depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()

        if not aligned_depth_frame or not color_frame:
            continue

        k = 0.5
        depth_image = k*depth_image + (1-k)*np.asanyarray(aligned_depth_frame.get_data()) * depth_scale
        color_image = np.asanyarray(color_frame.get_data())

        a, b, c, d = depth_image_ransac_processor.fit_transform(depth_image)
        plane = np.array([-a / c, -b / c, -d / c]).reshape(1, 1, 3)
        plane_image = (plane * indexes).sum(axis=2)

        m = numpy.where(np.abs(plane_image - depth_image) < 0.01)
        color_image[m] = color_image[m]/3 + [114, 114, 33]

        m = numpy.where(plane_image - depth_image <= -0.01)
        color_image[m] = color_image[m]/3 + [47, 31, 21]

        cv2.namedWindow('Align Example', cv2.WINDOW_NORMAL)
        cv2.imshow('Align Example', color_image)
        key = cv2.waitKey(1)
        if key & 0xFF == ord('q') or key == 27:
            cv2.destroyAllWindows()
            break
finally:
    pipeline.stop()
