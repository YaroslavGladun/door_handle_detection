import cv2
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
from common import CropImagesToAspectRatio, Point, Plane
from serialization import read_exr_points
from visualization_tools import draw_origin_on_image
from sklearn.linear_model import RANSACRegressor
from sklearn.linear_model import LinearRegression


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


depth = read_exr_points("EXR_RGBD/depth/300.exr")
rgb = cv2.imread("EXR_RGBD/rgb/300.jpg")

labler = LabelTestCase(rgb, depth)
affine = labler.start()
print(affine)

rgb_copy = rgb.copy()
draw_origin_on_image(rgb_copy, affine)
rgb_copy = cv2.resize(rgb_copy, (rgb_copy.shape[1]//2, rgb_copy.shape[0]//2))
cv2.imshow("WIN", rgb_copy)
cv2.waitKey()

R, t = affine[:-1, :-1], affine[:-1, 3]
points = depth.reshape(-1, 3)
# points = depth.reshape(-1, 3) - t
# points = np.dot(points, R)

point_cloud = o3d.geometry.PointCloud()
point_cloud.points = o3d.utility.Vector3dVector(points)


# Visualization function
def visualize_point_cloud(pcd):
    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window()

    # Add point cloud
    vis.add_geometry(pcd)
    coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2)
    coordinate_frame.transform(affine)
    vis.add_geometry(coordinate_frame)

    def build_move_function(i, increment):
        def move(*args, **kwargs):
            global affine

            last_affine = affine.copy()
            m = np.eye(4, 4, dtype=np.float32)
            m[i, 3] += increment
            affine = np.dot(affine, m)

            rgb_copy = rgb.copy()
            draw_origin_on_image(rgb_copy, affine)
            rgb_copy = cv2.resize(rgb_copy, (rgb_copy.shape[1] // 2, rgb_copy.shape[0] // 2))
            cv2.imshow("WIN", rgb_copy)
            cv2.waitKey(1)

            coordinate_frame.translate(affine[:-1, :-1] @ m[:-1, [3]])
            vis.update_geometry(coordinate_frame)
            vis.poll_events()
            vis.update_renderer()

        return move

    vis.register_key_callback(262, build_move_function(1, 0.005))
    vis.register_key_callback(263, build_move_function(1, -0.005))
    vis.register_key_callback(265, build_move_function(2, 0.005))
    vis.register_key_callback(264, build_move_function(2, -0.005))
    vis.register_key_callback(68, build_move_function(0, 0.005))
    vis.register_key_callback(65, build_move_function(0, -0.005))

    vis.run()
    vis.destroy_window()


visualize_point_cloud(point_cloud)
