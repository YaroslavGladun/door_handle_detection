import cv2
import numpy as np
import time
import pyrealsense2 as rs


net = cv2.dnn.readNetFromDarknet('/home/yaroslav/Downloads/yolo-obj.cfg', '/home/yaroslav/Downloads/yolo-obj.weights')
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)


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

align_to = rs.stream.color
align = rs.align(align_to)






classes = open('/home/yaroslav/Downloads/obj.names').read().strip().split('\n')
colors = np.random.randint(0, 255, size=(len(classes), 3), dtype='uint8')

try:
    while True:
        frames = pipeline.wait_for_frames()
        aligned_frames = align.process(frames)
        aligned_depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()

        if not aligned_depth_frame or not color_frame:
            continue

        img = np.asanyarray(color_frame.get_data())
        # img = cv2.imread('/home/yaroslav/Downloads/00c2c198953fc6d6.jpg')

        blob = cv2.dnn.blobFromImage(img, 1 / 255.0, (416, 416), swapRB=True, crop=False)
        net.setInput(blob)
        t0 = time.time()
        outputs = net.forward([])
        t = time.time()
        print('time=', t-t0)

        boxes = []
        confidences = []
        classIDs = []
        h, w = img.shape[:2]

        for output in outputs:
            for detection in output:
                scores = detection[5:]
                classID = np.argmax(scores)
                confidence = scores[classID]
                if confidence > 0.2:
                    box = detection[:4] * np.array([w, h, w, h])
                    (centerX, centerY, width, height) = box.astype("int")
                    x = int(centerX - (width / 2))
                    y = int(centerY - (height / 2))
                    box = [x, y, int(width), int(height)]
                    boxes.append(box)
                    confidences.append(float(confidence))
                    classIDs.append(classID)

        indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
        if len(indices) > 0:
            for i in indices.flatten():
                (x, y) = (boxes[i][0], boxes[i][1])
                (w, h) = (boxes[i][2], boxes[i][3])
                color = [int(c) for c in colors[classIDs[i]]]
                cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
                text = "{}: {:.4f}".format(classes[classIDs[i]], confidences[i])
                cv2.putText(img, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        cv2.namedWindow('Align Example', cv2.WINDOW_NORMAL)
        cv2.imshow('Align Example', img)
        key = cv2.waitKey(1)
        if key & 0xFF == ord('q') or key == 27:
            cv2.destroyAllWindows()
            break
        # cv2.imshow('window', img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()


        # cv2.namedWindow('Align Example', cv2.WINDOW_NORMAL)
        # cv2.imshow('Align Example', color_image)
        # key = cv2.waitKey(1)
        # if key & 0xFF == ord('q') or key == 27:
        #     cv2.destroyAllWindows()
        #     break
finally:
    pipeline.stop()

# img = cv.imread('/home/yaroslav/Downloads/00c2c198953fc6d6.jpg')
# cv.imshow('window',  img)
# cv.waitKey(1)
#
# classes = open('/home/yaroslav/Downloads/obj.names').read().strip().split('\n')
# colors = np.random.randint(0, 255, size=(len(classes), 3), dtype='uint8')
#
# net = cv.dnn.readNetFromDarknet('/home/yaroslav/Downloads/yolo-obj.cfg', '/home/yaroslav/Downloads/yolo-obj.weights')
# net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)
# blob = cv.dnn.blobFromImage(img, 1/255.0, (416, 416), swapRB=True, crop=False)
# r = blob[0, 0, :, :]
#
# cv.imshow('blob', r)
# text = f'Blob shape={blob.shape}'
# cv.displayOverlay('blob', text)
# cv.waitKey(1)
#
# net.setInput(blob)
# t0 = time.time()
# outputs = net.forward([])
# t = time.time()
# print('time=', t-t0)
#
# print(len(outputs))
# for out in outputs:
#     print(out.shape)
#
# def trackbar2(x):
#     confidence = x/100
#     r = r0.copy()
#     for output in np.vstack(outputs):
#         if output[4] > confidence:
#             x, y, w, h = output[:4]
#             p0 = int((x-w/2)*416), int((y-h/2)*416)
#             p1 = int((x+w/2)*416), int((y+h/2)*416)
#             cv.rectangle(r, p0, p1, 1, 1)
#     cv.imshow('blob', r)
#     text = f'Bbox confidence={confidence}'
#     cv.displayOverlay('blob', text)
#
# r0 = blob[0, 0, :, :]
# r = r0.copy()
# cv.imshow('blob', r)
# cv.createTrackbar('confidence', 'blob', 50, 101, trackbar2)
# trackbar2(50)
#
# boxes = []
# confidences = []
# classIDs = []
# h, w = img.shape[:2]
#
# for output in outputs:
#     for detection in output:
#         scores = detection[5:]
#         classID = np.argmax(scores)
#         confidence = scores[classID]
#         if confidence > 0.5:
#             box = detection[:4] * np.array([w, h, w, h])
#             (centerX, centerY, width, height) = box.astype("int")
#             x = int(centerX - (width / 2))
#             y = int(centerY - (height / 2))
#             box = [x, y, int(width), int(height)]
#             boxes.append(box)
#             confidences.append(float(confidence))
#             classIDs.append(classID)
#
# indices = cv.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
# if len(indices) > 0:
#     for i in indices.flatten():
#         (x, y) = (boxes[i][0], boxes[i][1])
#         (w, h) = (boxes[i][2], boxes[i][3])
#         color = [int(c) for c in colors[classIDs[i]]]
#         cv.rectangle(img, (x, y), (x + w, y + h), color, 2)
#         text = "{}: {:.4f}".format(classes[classIDs[i]], confidences[i])
#         cv.putText(img, text, (x, y - 5), cv.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
#
# cv.imshow('window', img)
# cv.waitKey(0)
# cv.destroyAllWindows()
