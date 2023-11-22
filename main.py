import open3d as o3d
import numpy as np

# Example: Create a random point cloud
points = np.random.rand(100, 3)  # 100 points in 3D
point_cloud = o3d.geometry.PointCloud()
point_cloud.points = o3d.utility.Vector3dVector(points)


# Visualization function
def visualize_point_cloud(pcd):
    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window()

    # Add point cloud
    vis.add_geometry(pcd)

    # Define key callbacks to move the point cloud
    def move_along_x(*args, **kwargs):
        for point in pcd.points:
            point[0] += 0.005
        vis.update_geometry(pcd)
        vis.poll_events()
        vis.update_renderer()

    def move_along_y(*args, **kwargs):
        for point in pcd.points:
            point[1] += 0.005
        vis.update_geometry(pcd)
        vis.poll_events()
        vis.update_renderer()

    def move_along_z(*args, **kwargs):
        for point in pcd.points:
            point[2] += 0.005
        vis.update_geometry(pcd)
        vis.poll_events()
        vis.update_renderer()

    # Assign keys for the sliders
    vis.register_key_callback(262, move_along_x)  # Right arrow key
    vis.register_key_callback(263, move_along_y)  # Left arrow key
    vis.register_key_callback(264, move_along_z)  # Down arrow key

    vis.run()
    vis.destroy_window()


visualize_point_cloud(point_cloud)
