import cv2
import numpy as np
import matplotlib.pyplot as plt
from common import CropImagesToAspectRatio
from serialization import read_exr_points
from sklearn.linear_model import RANSACRegressor
from sklearn.linear_model import LinearRegression


class Point(np.ndarray):

    @staticmethod
    def normalized(a, b, c):
        p = Point([a, b, c])
        return p / np.linalg.norm(p)

    def __new__(cls, input_array):
        if len(input_array) != 3:
            raise ValueError("Point must have a size of 3")

        obj = np.asarray(input_array).view(cls)
        return obj


class Plane(np.ndarray):

    @staticmethod
    def normalized(a, b, c, d):
        abc_norm = np.linalg.norm([a, b, c])
        a, b, c, d = a / abc_norm, b / abc_norm, c / abc_norm, d / abc_norm
        return Plane([a, b, c, d])

    def __new__(cls, input_array):
        if len(input_array) != 4:
            raise ValueError("Plane must have a size of 4")

        obj = np.asarray(input_array).view(cls)
        return obj

    def norm(self) -> Point:
        a, b, c, _ = self
        return Point.normalized(a, b, c)


class LabelTestCase:
    GRID_SIZE = 7
    FX, FY = 181.45, 181.45
    CX, CY = 96.7, 96.7

    def __init__(self, rgb: np.ndarray, depth: np.ndarray):
        self.__plane_grid = np.zeros((7, 7), dtype=np.bool_)
        self.__handle_end_show = None
        self.__handle_begin_show = None
        self.__plane_grid_show_ax2 = None
        self.__plane_grid_show_ax1 = None
        self.__fig = None
        self.__ax1 = None
        self.__ax2 = None
        self.__depth = CropImagesToAspectRatio.crop_images(depth, 1, 1)[0]
        self.__cropped_rgb = CropImagesToAspectRatio.crop_images(rgb, 1, 1)[0]
        self.__rgb = cv2.resize(self.__cropped_rgb, self.__depth.shape[:-1])
        self.__cell_size = self.__depth.shape[0] / self.GRID_SIZE
        self.__plane = None

    def start(self):
        self.__fig, (self.__ax1, self.__ax2) = plt.subplots(1, 2)
        self.__ax1.imshow(self.__depth[..., 2])
        self.__ax2.imshow(self.__rgb[:, :, ::-1])
        self.draw_grid()

        self.__plane_grid_show_ax1 = self.__ax1.imshow(
            np.zeros(self.__depth.shape[:-1], dtype=np.float32),
            cmap='gray',
            alpha=0.5 * cv2.resize(self.__plane_grid.astype(np.float32), self.__depth.shape[:-1],
                                   interpolation=cv2.INTER_NEAREST))
        self.__plane_grid_show_ax2 = self.__ax2.imshow(
            np.zeros(self.__rgb.shape[:-1], dtype=np.float32),
            cmap='gray',
            alpha=0.5 * cv2.resize(self.__plane_grid.astype(np.float32), self.__rgb.shape[:-1],
                                   interpolation=cv2.INTER_NEAREST))
        self.__fig.canvas.mpl_connect('button_release_event', self.on_release)

        self.__handle_begin_show = None
        self.__handle_end_show = None

        self.__fig.canvas.mpl_connect('key_release_event', self.on_key_release)

        plt.show()

        return self.get_affine()

    def on_release(self, event):
        i, j = int(event.ydata // self.__cell_size), int(event.xdata // self.__cell_size)
        self.__plane_grid[i, j] = not self.__plane_grid[i, j]
        depth_mask = cv2.resize(
            self.__plane_grid.astype(np.int32), self.__depth.shape[:-1], interpolation=cv2.INTER_NEAREST).astype(
            np.bool_)
        points = self.__depth[depth_mask]
        ransac = RANSACRegressor(LinearRegression())
        ransac.fit(points[:, [0, 1]], points[:, 2])
        (a, b), c = ransac.estimator_.coef_, -1
        d = ransac.estimator_.intercept_
        self.__plane = Plane.normalized(a, b, c, d)
        print(f"Plane equation: {a}x + {b}y + {c}z + {d} = 0")
        self.__plane_grid_show_ax1.set_alpha(0.5 * depth_mask.astype(np.float32))
        self.__plane_grid_show_ax2.set_alpha(
            0.5 * cv2.resize(self.__plane_grid.astype(np.float32), self.__rgb.shape[:-1],
                             interpolation=cv2.INTER_NEAREST))
        self.__fig.canvas.draw()
        print(f'Mouse released at ({event.xdata}, {event.ydata})')

    def on_key_release(self, event):
        global new_points
        if event.key == 'a':
            if event.xdata is None or event.ydata is None:
                return
            i, j = event.xdata, event.ydata
            if self.__handle_begin_show is None:
                self.__handle_begin_show = self.__ax2.scatter([i], [j], marker='x', s=100, color="green")
            else:
                self.__handle_begin_show.set_offsets([[i, j]])
            self.__fig.canvas.draw()
        if event.key == 'b':
            if event.xdata is None or event.ydata is None:
                return
            i, j = event.xdata, event.ydata
            if self.__handle_end_show is None:
                self.__handle_end_show = self.__ax2.scatter([i], [j], marker='x', s=100, color="red")
            else:
                self.__handle_end_show.set_offsets([[i, j]])
            self.__fig.canvas.draw()

    def draw_grid(self):
        for l in np.arange(0, self.__depth.shape[0], self.__cell_size):
            self.__ax1.axvline(l, color='black', linestyle='-')
            self.__ax1.axhline(l, color='black', linestyle='-')
            self.__ax2.axvline(l, color='black', linestyle='-')
            self.__ax2.axhline(l, color='black', linestyle='-')

    def get_handle_projection(self):
        assert self.FX == self.FY
        assert self.CX == self.CY

        def g(u, v):
            p, f = self.CX, self.FX
            a, b, c, d = self.__plane
            multiplier = d / (a * p - a * u + b * p - b * v - c * f)
            x = multiplier * (u - p)
            y = multiplier * (v - p)
            z = multiplier * f
            return Point([x, y, z])

        self.__p1 = g(*self.__handle_begin_show.get_offsets().reshape(-1))
        self.__p2 = g(*self.__handle_end_show.get_offsets().reshape(-1))
        return self.__p1, self.__p2

    def get_affine(self):
        _, _ = self.get_handle_projection()
        a = Point.normalized(*(self.__p2 - self.__p1))
        b = -self.__plane.norm()
        c = np.cross(a, b)
        p = self.__p1
        affine_matrix = np.array([
            [a[0], b[0], c[0], p[0]],
            [a[1], b[1], c[1], p[1]],
            [a[2], b[2], c[2], p[2]],
            [0, 0, 0, 1]
        ])
        return affine_matrix

    @staticmethod
    def draw_origin_on_image(image: np.ndarray, pose: np.ndarray):
        # fx, fy = LabelTestCase.FX, LabelTestCase.FY
        fx, fy = 2 * (1421.0684814453125,)
        # px, py = LabelTestCase.CX, LabelTestCase.CY
        px, py = 724.485107421875, 965.93603515625
        a0 = pose[:-1, 3]
        ax = np.dot(pose, np.array([0.02, 0, 0, 1]).reshape(-1, 1))[:-1]
        ay = np.dot(pose, np.array([0, 0.02, 0, 1]).reshape(-1, 1))[:-1]
        az = np.dot(pose, np.array([0, 0, 0.02, 1]).reshape(-1, 1))[:-1]

        a0_cv_pt = (int(fx * a0[0] / a0[2] + px), int(fy * a0[1] / a0[2] + py))
        ax_cv_pt = (int(fx * ax[0] / ax[2] + px), int(fy * ax[1] / ax[2] + py))
        ay_cv_pt = (int(fx * ay[0] / ay[2] + px), int(fy * ay[1] / ay[2] + py))
        az_cv_pt = (int(fx * az[0] / az[2] + px), int(fy * az[1] / az[2] + py))

        cv2.line(image, a0_cv_pt, ax_cv_pt, (0, 0, 255), 3)  # Red for X-axis
        cv2.line(image, a0_cv_pt, ay_cv_pt, (0, 255, 0), 3)  # Green for Y-axis
        cv2.line(image, a0_cv_pt, az_cv_pt, (255, 0, 0), 3)  # Blue for Z-axis

        return image


depth = read_exr_points("/Users/hladunyaroslav/Desktop/door_handle_labeling/EXR_RGBD/depth/300.exr")
rgb = cv2.imread("/Users/hladunyaroslav/Desktop/door_handle_labeling/EXR_RGBD/rgb/300.jpg")

labler = LabelTestCase(rgb, depth)
affine = labler.start()
print(affine)

LabelTestCase.draw_origin_on_image(rgb, affine)
cv2.imshow("WIN", rgb)
cv2.waitKey()

R, t = affine[:-1, :-1], affine[:-1, 3]
points = depth.reshape(-1, 3) - t
points = np.dot(points, R)
# points = points[np.linalg.norm(points, axis=1) < 0.30]
# points = np.dot(points, affine)[:, :-1]
import matplotlib as mpl
import open3d as o3d


# pcd = o3d.geometry.PointCloud()
# pcd.points = o3d.utility.Vector3dVector(points)
# mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])
# o3d.visualization.draw_geometries([pcd, mesh_frame])


class CustomVisualization:
    def __init__(self, point_cloud, mesh_frame):
        self.window = o3d.visualization.VisualizerWithKeyCallback()
        self.window.create_window()
        self.point_cloud = point_cloud
        self.mesh_frame = mesh_frame
        self.window.add_geometry(self.point_cloud)
        self.window.add_geometry(self.mesh_frame)

        # Add GUI sliders
        self.create_sliders()

    def create_sliders(self):
        # Create sliders and set callbacks
        self.x_slider = self.window.create_slider("X", 0, 100, 50)
        self.x_slider.set_on_value_change_callback(self.update_frame_position)

        self.y_slider = self.window.create_slider("Y", 0, 100, 50)
        self.y_slider.set_on_value_change_callback(self.update_frame_position)

        self.z_slider = self.window.create_slider("Z", 0, 100, 50)
        self.z_slider.set_on_value_change_callback(self.update_frame_position)

    def update_frame_position(self):
        # Get slider values and update mesh frame position
        x = self.x_slider.get_value() / 100.0
        y = self.y_slider.get_value() / 100.0
        z = self.z_slider.get_value() / 100.0
        self.mesh_frame.translate([x, y, z], relative=False)
        self.window.update_geometry(self.mesh_frame)
        self.window.poll_events()
        self.window.update_renderer()

    def run(self):
        self.window.run()
        self.window.destroy_window()


# Your point cloud data
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points)

# Create mesh frame
mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])

# Create and run custom visualization
custom_vis = CustomVisualization(pcd, mesh_frame)
custom_vis.run()
