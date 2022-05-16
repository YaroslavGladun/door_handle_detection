import math
import time
import cv2
import numpy as np
import pyrealsense2 as rs
from door_handle_detection_utils import RANSACProcessor


class AppState:

    def __init__(self):
        self.WIN_NAME = 'RealSense'
        self.pitch, self.yaw = math.radians(-10), math.radians(-15)
        self.translation = np.array([0, 0, -1], dtype=np.float32)
        self.distance = 2
        self.prev_mouse = 0, 0
        self.mouse_btns = [False, False, False]
        self.paused = False
        self.decimate = 1
        self.scale = True
        self.color = True

    def reset(self):
        self.pitch, self.yaw, self.distance = 0, 0, 2
        self.translation[:] = 0, 0, -1

    @property
    def rotation(self):
        Rx, _ = cv2.Rodrigues((self.pitch, 0, 0))
        Ry, _ = cv2.Rodrigues((0, self.yaw, 0))
        return np.dot(Ry, Rx).astype(np.float32)

    @property
    def pivot(self):
        return self.translation + np.array((0, 0, self.distance), dtype=np.float32)


class App:

    def __init__(self, window_width, window_height):
        self.state = AppState()

        cv2.namedWindow(self.state.WIN_NAME, cv2.WINDOW_AUTOSIZE)
        cv2.resizeWindow(self.state.WIN_NAME, window_width, window_height)
        cv2.setMouseCallback(self.state.WIN_NAME, self.mouse_cb)

        self.out = np.empty((window_height, window_width, 3), dtype=np.uint8)

    def mouse_cb(self, event, x, y, flags, params):

        if event == cv2.EVENT_LBUTTONDOWN:
            self.state.mouse_btns[0] = True

        if event == cv2.EVENT_LBUTTONUP:
            self.state.mouse_btns[0] = False

        if event == cv2.EVENT_RBUTTONDOWN:
            self.state.mouse_btns[1] = True

        if event == cv2.EVENT_RBUTTONUP:
            self.state.mouse_btns[1] = False

        if event == cv2.EVENT_MBUTTONDOWN:
            self.state.mouse_btns[2] = True

        if event == cv2.EVENT_MBUTTONUP:
            self.state.mouse_btns[2] = False

        if event == cv2.EVENT_MOUSEMOVE:

            h, w = self.out.shape[:2]
            dx, dy = x - self.state.prev_mouse[0], y - self.state.prev_mouse[1]

            if self.state.mouse_btns[0]:
                self.state.yaw += float(dx) / w * 2
                self.state.pitch -= float(dy) / h * 2

            elif self.state.mouse_btns[1]:
                dp = np.array((dx / w, dy / h, 0), dtype=np.float32)
                self.state.translation -= np.dot(self.state.rotation, dp)

            elif self.state.mouse_btns[2]:
                dz = math.sqrt(dx ** 2 + dy ** 2) * math.copysign(0.01, -dy)
                self.state.translation[2] += dz
                self.state.distance -= dz

        if event == cv2.EVENT_MOUSEWHEEL:
            dz = math.copysign(0.1, flags)
            self.state.translation[2] += dz
            self.state.distance -= dz

        self.state.prev_mouse = (x, y)

    def line3d(self, pt1, pt2, color=(0x80, 0x80, 0x80), thickness=1):
        """draw a 3d line from pt1 to pt2"""
        p0 = self.project(pt1.reshape(-1, 3))[0]
        p1 = self.project(pt2.reshape(-1, 3))[0]
        if np.isnan(p0).any() or np.isnan(p1).any():
            return
        p0 = tuple(p0.astype(int))
        p1 = tuple(p1.astype(int))
        rect = (0, 0, self.out.shape[1], self.out.shape[0])
        inside, p0, p1 = cv2.clipLine(rect, p0, p1)
        if inside:
            cv2.line(self.out, p0, p1, color, thickness, cv2.LINE_AA)

    def grid(self, pos, rotation=np.eye(3), size=1, n=10, color=(0x80, 0x80, 0x80)):
        """draw a grid on xz plane"""
        pos = np.array(pos)
        s = size / float(n)
        s2 = 0.5 * size
        for i in range(0, n + 1):
            x = -s2 + i * s
            self.line3d(self.view(pos + np.dot((x, 0, -s2), rotation)),
                        self.view(pos + np.dot((x, 0, s2), rotation)), color)
        for i in range(0, n + 1):
            z = -s2 + i * s
            self.line3d(self.view(pos + np.dot((-s2, 0, z), rotation)),
                        self.view(pos + np.dot((s2, 0, z), rotation)), color)

    def axes(self, pos, rotation=np.eye(3), size=0.075, thickness=2):
        """draw 3d axes"""
        self.line3d(pos, pos +
                   np.dot((0, 0, size), rotation), (0xff, 0, 0), thickness)
        self.line3d(pos, pos +
                   np.dot((0, size, 0), rotation), (0, 0xff, 0), thickness)
        self.line3d(pos, pos +
                   np.dot((size, 0, 0), rotation), (0, 0, 0xff), thickness)

    def frustum(self, intrinsics, color=(0x40, 0x40, 0x40)):
        """draw camera's frustum"""
        orig = self.view(np.zeros(3))
        w, h = intrinsics.width, intrinsics.height

        for d in range(1, 6, 2):
            def get_point(x, y):
                p = rs.rs2_deproject_pixel_to_point(intrinsics, [x, y], d)
                self.line3d(orig, self.view(p), color)
                return p

            top_left = get_point(0, 0)
            top_right = get_point(w, 0)
            bottom_right = get_point(w, h)
            bottom_left = get_point(0, h)

            self.line3d(self.view(top_left), self.view(top_right), color)
            self.line3d(self.view(top_right), self.view(bottom_right), color)
            self.line3d(self.view(bottom_right), self.view(bottom_left), color)
            self.line3d(self.view(bottom_left), self.view(top_left), color)

    def project(self, v):
        """project 3d vector array to 2d"""
        h, w = self.out.shape[:2]
        view_aspect = float(h) / w

        # ignore divide by zero for invalid depth
        with np.errstate(divide='ignore', invalid='ignore'):
            proj = v[:, :-1] / v[:, -1, np.newaxis] * \
                   (w * view_aspect, h) + (w / 2.0, h / 2.0)

        # near clipping
        znear = 0.03
        proj[v[:, 2] < znear] = np.nan
        return proj

    def view(self, v: np.ndarray):
        """apply view transformation on vector array"""
        return np.dot(v - self.state.pivot, self.state.rotation) + self.state.pivot - self.state.translation

    def pointcloud(self, out, verts, texcoords, color, painter=True):
        """draw point cloud with optional painter's algorithm"""
        if painter:
            # Painter's algo, sort points from back to front

            # get reverse sorted indices by z (in view-space)
            # https://gist.github.com/stevenvo/e3dad127598842459b68
            v = app.view(verts)
            s = v[:, 2].argsort()[::-1]
            proj = app.project(v[s])
        else:
            proj = app.project(app.view(verts))

        if app.state.scale:
            proj *= 0.5 ** app.state.decimate

        h, w = out.shape[:2]

        # proj now contains 2d image coordinates
        j, i = proj.astype(np.uint32).T

        # create a mask to ignore out-of-bound indices
        im = (i >= 0) & (i < h)
        jm = (j >= 0) & (j < w)
        m = im & jm

        cw, ch = color.shape[:2][::-1]
        if painter:
            # sort texcoord with same indices as above
            # texcoords are [0..1] and relative to top-left pixel corner,
            # multiply by size and add 0.5 to center
            v, u = (texcoords[s] * (cw, ch) + 0.5).astype(np.uint32).T
        else:
            v, u = (texcoords * (cw, ch) + 0.5).astype(np.uint32).T
        # clip texcoords to image
        np.clip(u, 0, ch - 1, out=u)
        np.clip(v, 0, cw - 1, out=v)

        # perform uv-mapping
        out[i[m], j[m]] = color[u[m], v[m]]


# Configure depth and color streams
pipeline = rs.pipeline()
config = rs.config()

pipeline_wrapper = rs.pipeline_wrapper(pipeline)
pipeline_profile = config.resolve(pipeline_wrapper)
device = pipeline_profile.get_device()

found_rgb = False
for s in device.sensors:
    if s.get_info(rs.camera_info.name) == 'RGB Camera':
        found_rgb = True
        break
if not found_rgb:
    print("The demo requires Depth camera with Color sensor")
    exit(0)

config.enable_stream(rs.stream.depth, rs.format.z16, 30)
config.enable_stream(rs.stream.color, rs.format.bgr8, 30)

# Start streaming
pipeline.start(config)

# Get stream profile and camera intrinsics
profile = pipeline.get_active_profile()
depth_profile = rs.video_stream_profile(profile.get_stream(rs.stream.depth))
depth_intrinsics = depth_profile.get_intrinsics()
w, h = depth_intrinsics.width, depth_intrinsics.height

app = App(w, h)

# Processing blocks
pc = rs.pointcloud()
decimate = rs.decimation_filter()
decimate.set_option(rs.option.filter_magnitude, 2 ** app.state.decimate)
colorizer = rs.colorizer()


ransac_processor = RANSACProcessor(1000, 500)
plane = np.array([0, 0, 0, 0])
while True:
    # Grab camera data
    if not app.state.paused:
        # Wait for a coherent pair of frames: depth and color
        frames = pipeline.wait_for_frames()

        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()

        depth_frame = decimate.process(depth_frame)

        # Grab new intrinsics (may be changed by decimation)
        depth_intrinsics = rs.video_stream_profile(
            depth_frame.profile).get_intrinsics()
        w, h = depth_intrinsics.width, depth_intrinsics.height

        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        depth_colormap = np.asanyarray(
            colorizer.colorize(depth_frame).get_data())

        if app.state.color:
            mapped_frame, color_source = color_frame, color_image
        else:
            mapped_frame, color_source = depth_frame, depth_colormap

        points = pc.calculate(depth_frame)
        pc.map_to(mapped_frame)

        # Pointcloud data to arrays
        v, t = points.get_vertices(), points.get_texture_coordinates()
        verts = np.asanyarray(v).view(np.float32).reshape(-1, 3)  # xyz
        plane = ransac_processor.fit_transform(verts)
        texcoords = np.asanyarray(t).view(np.float32).reshape(-1, 2)  # uv

    # Render
    now = time.time()

    app.out.fill(0)

    app.grid((0, 0.5, 1), size=1, n=10)
    app.frustum(depth_intrinsics)
    app.axes(app.view([0, 0, 0]), app.state.rotation, size=0.1, thickness=1)
    app.line3d(app.view([0, 0, 0]), app.view(10 * plane[:-1]), thickness=2)

    if not app.state.scale or app.out.shape[:2] == (h, w):
        app.pointcloud(verts, texcoords, color_source)
    else:
        tmp = np.zeros((h, w, 3), dtype=np.uint8)
        app.pointcloud(tmp, verts, texcoords, color_source)
        tmp = cv2.resize(
            tmp, app.out.shape[:2][::-1], interpolation=cv2.INTER_NEAREST)
        np.putmask(app.out, tmp > 0, tmp)

    if any(app.state.mouse_btns):
        app.axes(app.view(app.state.pivot), app.state.rotation, thickness=4)

    dt = time.time() - now

    cv2.setWindowTitle(
        app.state.WIN_NAME, "RealSense (%dx%d) %dFPS (%.2fms) %s" %
                            (w, h, 1.0 / dt, dt * 1000, "PAUSED" if app.state.paused else ""))

    cv2.imshow(app.state.WIN_NAME, app.out)
    key = cv2.waitKey(1)

    if key == ord("r"):
        app.state.reset()

    if key == ord("p"):
        app.state.paused ^= True

    if key == ord("d"):
        app.state.decimate = (app.state.decimate + 1) % 3
        decimate.set_option(rs.option.filter_magnitude, 2 ** app.state.decimate)

    if key == ord("z"):
        app.state.scale ^= True

    if key == ord("c"):
        app.state.color ^= True

    if key in (27, ord("q")) or cv2.getWindowProperty(app.state.WIN_NAME, cv2.WND_PROP_AUTOSIZE) < 0:
        break

pipeline.stop()
