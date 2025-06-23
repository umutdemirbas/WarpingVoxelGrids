import open3d as o3d
import numpy as np
from numpy.ma.extras import unique
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import lsqr
import sys
import os
import open3d.visualization as vis
from open3d import data
#print(hasattr(vis, "gui"), hasattr(vis, "rendering"))
import copy
from scipy.spatial.transform import Rotation as R
import random
import cv2

"""
Combined approach using mesh ray casting and depth projection to get 3D points from RGB-D frames.
-Depth projection uses the distance info from depth images and projects it into 3D space using camera intrinsics and
 extrinsics.
-Mesh raycasting ignores the depth images. Instead, fires a ray from the camera through the pixel and computes where it
 intersects with the triangle mesh.
 
Results of both are saved in a combined text file with the following format: {frame_id} {u} {v} {xd:.6f} {yd:.6f} {zd:.6f} {xm:.6f} {ym:.6f} {zm:.6f}
Depth projections can be used as a fallback for the mesh raycasting if no intersection is found. (Incomplete meshes)
It can also be used to validate the mesh raycasting results.

"""

# --- CONFIG ---
rgb_dir = "./data/rgbd_frames/rgb/"
depth_dir = "./data/rgbd_frames/depth/"
pose_file = "./data/standard_trajectory_no_loop.txt"  # Format: frame_id tx ty tz qx qy qz qw
fx, fy, cx, cy = 377.535257164, 377.209841379, 328.193371286, 240.426878936  # [fu, fv, cu, cv]
depth_scale = 1000.0  # adjust based on your depth units --> 1.0 if depth data is in meters, 1000.0 if mm
num_samples = 100

# --- Load poses ---
# same as before
def load_poses_with_frame_ids(path):
    poses = {} # Dictionary to hold poses with frame_id as key
    with open(path, 'r') as f:
        lines = f.readlines()
        for i, line in enumerate(lines):
            parts = line.strip().split()
            t = np.array([float(x) for x in parts[1:4]])
            q = np.array([float(x) for x in parts[4:]])
            rot = R.from_quat(q).as_matrix()
            T = np.eye(4)
            T[:3, :3] = rot
            T[:3, 3] = t
            poses[i] = T  # Use frame index (0, 1, ...) as frame_id
    return poses

poses = load_poses_with_frame_ids(pose_file)
#print(poses.items())

# --- Load mesh and setup raycasting ---
# Replace with your mesh path
mesh = o3d.io.read_triangle_mesh("data/output_mesh_from_tsdf.ply")
mesh.compute_vertex_normals()
scene = o3d.t.geometry.RaycastingScene()
scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
# camera intrinsics
K = np.array([[fx, 0, cx],
              [0, fy, cy],
              [0,  0, 1]])
K_inv = np.linalg.inv(K)


# --- Process frames ---
output = []

for frame_id, pose in poses.items():
    filename = f"frame_{frame_id:05d}.png"
    rgb_path = os.path.join(rgb_dir, filename)
    depth_path = os.path.join(depth_dir, filename)
    if not os.path.exists(rgb_path) or not os.path.exists(depth_path):
        print(f"Skipping frame {frame_id:05d}: RGB or depth image not found.")
        continue

    depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
    h, w = depth.shape

    # randomly sample pixels from the depth image
    sampled_pixels = [(random.randint(0, w - 1), random.randint(0, h - 1)) for _ in range(num_samples)]
    #print("sampled: ",sampled_pixels)
    # Project the pixels into 3D space using the camera pose and depth values
    for u, v in sampled_pixels:
        # first project from pixel coordinates to camera coordinates using camera intrinsics
        Z = depth[v, u] / depth_scale
        if Z <= 0:
            continue
        X = (u - cx) * Z / fx
        Y = (v - cy) * Z / fy
        # then project from camera coordinates to world coordinates using the pose (extrinsics)
        point_cam = np.array([X, Y, Z, 1.0])
        point_world_depth = pose @ point_cam
        # ----------------------------------------------------
        # Raycasting
        pixel_hom = np.array([u, v, 1.0])
        ray_cam = K_inv @ pixel_hom
        ray_cam /= np.linalg.norm(ray_cam)
        origin = pose[:3, 3]
        direction = pose[:3, :3] @ ray_cam

        ray = o3d.core.Tensor([[origin[0], origin[1], origin[2],
                                direction[0], direction[1], direction[2]]],
                              dtype=o3d.core.Dtype.Float32)
        result = scene.cast_rays(ray)
        t_hit = result['t_hit'].numpy()[0]

        if np.isfinite(t_hit):
            hit_point = origin + t_hit * direction
            output.append({
                "frame": frame_id,
                "pixel": [u, v],
                "depth_point": point_world_depth[:3].tolist(),
                "mesh_hit": hit_point.tolist()
            })
        # ----------------------------------------------------

# --- Save combined results as .txt ---
with open("raycast_combined_points_no_loop.txt", "w") as f:
    for entry in output:
        frame_id = entry["frame"]
        u, v = entry["pixel"]
        xd, yd, zd = entry["depth_point"]
        xm, ym, zm = entry["mesh_hit"]
        f.write(f"{frame_id} {u} {v} "
                f"{xd:.6f} {yd:.6f} {zd:.6f} "
                f"{xm:.6f} {ym:.6f} {zm:.6f}\n")
