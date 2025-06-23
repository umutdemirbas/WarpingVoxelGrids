import open3d as o3d
from open3d import data
import open3d.core as o3c
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KDTree
import time

# =====================
# Utility Functions
# =====================

def sample_surface_points(mesh, number_of_points=10000):
    """
    Uniformly sample points from the surface of a mesh.
    Returns an Open3D PointCloud.
    """
    return mesh.sample_points_uniformly(number_of_points)


def compute_surface_distance(src_pc, target_mesh):
    """
    Compute the unsigned distance from each point in src_pc to the target_mesh surface using Open3D's raycasting.
    Returns a numpy array of distances.
    """
    target_t = o3d.t.geometry.TriangleMesh.from_legacy(target_mesh)
    scene = o3d.t.geometry.RaycastingScene()
    _ = scene.add_triangles(target_t)

    points = np.asarray(src_pc.points)
    query = o3c.Tensor(points, dtype=o3c.float32)
    dists = scene.compute_distance(query).numpy()

    return dists


def compute_fscore(pcd_pred, pcd_gt, threshold):
    """
    Compute the F-score, precision, and recall between two point clouds at a given threshold.
    """
    pred_points = np.asarray(pcd_pred.points)
    gt_points = np.asarray(pcd_gt.points)

    tree_gt = KDTree(gt_points)
    dist_pred_to_gt, _ = tree_gt.query(pred_points, k=1)
    precision = np.mean(dist_pred_to_gt < threshold)

    tree_pred = KDTree(pred_points)
    dist_gt_to_pred, _ = tree_pred.query(gt_points, k=1)
    recall = np.mean(dist_gt_to_pred < threshold)

    if precision + recall == 0:
        fscore = 0.0
    else:
        fscore = 2 * precision * recall / (precision + recall)

    return precision, recall, fscore

# =====================
# Parameters
# =====================
voxel_size = 0.25
truncation_distance = 0.1
ball_radius_mx = 0.4

print("Testing mesh in Open3D...")
# Reading and computing the mesh
# armadillo_mesh = o3d.data.BunnyMesh()
mesh = o3d.io.read_triangle_mesh("mesh_edit/edited_mesh.ply")
mesh.compute_vertex_normals()
vertices = np.asarray(mesh.vertices)
# colors = plt.cm.viridis((vertices[:, 1] - vertices[:, 1].min()) / (vertices[:, 1].ptp()))[:, :3]
# mesh.vertex_colors = o3d.utility.Vector3dVector(colors)
o3d.visualization.draw_geometries([mesh], window_name="Original Mesh")

# Convert the mesh to a tensor mesh for raycasting
mesh_t = o3d.t.geometry.TriangleMesh.from_legacy(mesh)

# Set up the raycasting scene
scene = o3d.t.geometry.RaycastingScene()
scene.add_triangles(mesh_t)

print("Voxelization")
voxel_start = time.time()
# Voxelize the mesh
voxel_grid = o3d.geometry.VoxelGrid.create_from_triangle_mesh(mesh, voxel_size)
print(f"Voxelization time: {time.time() - voxel_start:.4f} seconds")
o3d.visualization.draw_geometries([voxel_grid])

# Prepare for TSDF computation
tsdf_values = {}
points_from_voxel = []
voxelsize = voxel_grid.voxel_size
grid_origin = voxel_grid.origin

# Get vertex positions and colors for color interpolation
vertex_positions = np.asarray(mesh.vertices)
vertex_colors = np.asarray(mesh.vertex_colors)

# tsdf_start: timing the TSDF computation
tsdf_start = time.time()

# Compute TSDF values for each voxel center
for voxel in voxel_grid.get_voxels():
    voxel_index = voxel.grid_index
    voxel_center = voxel_grid.get_voxel_center_coordinate(voxel_index)
    # points_from_voxel: collect voxel centers for point cloud reconstruction
    points_from_voxel.append(voxel_center)

    # Compute signed distance from voxel center to mesh surface
    signed_distance = scene.compute_signed_distance(
        o3c.Tensor([voxel_center], dtype=o3c.float32)).item()

    # Truncate and normalize TSDF values
    if signed_distance > truncation_distance:
        tsdf_values[tuple(voxel_index)] = 1.0
    elif signed_distance < -truncation_distance:
        tsdf_values[tuple(voxel_index)] = -1.0
    else:
        tsdf_values[tuple(voxel_index)] = signed_distance / truncation_distance

    # Color assignment (optional): find nearest vertex and assign its color
    # dists = np.linalg.norm(vertex_positions - voxel_center, axis=1)
    # nearest_idx = np.argmin(dists)
    # voxel_colors[tuple(voxel_index)] = vertex_colors[nearest_idx]

print(f"TSDF and color assignment time: {time.time() - tsdf_start:.4f} seconds")

# =====================
# Point Cloud and Mesh Reconstruction
# =====================
point_cloud_start = time.time()
# Create a point cloud from voxel centers
point_cloud = o3d.geometry.PointCloud()
point_cloud.points = o3d.utility.Vector3dVector(points_from_voxel)
point_cloud.estimate_normals()
o3d.geometry.PointCloud.orient_normals_consistent_tangent_plane(point_cloud, 50)
print(f"Point cloud creation time: {time.time() - point_cloud_start:.4f} seconds")
# o3d.visualization.draw_geometries([point_cloud], width=1200, height=800)

# Estimate radius for ball pivoting
ball_pivoting_time = time.time()
avg_dist = np.mean(point_cloud.compute_nearest_neighbor_distance())
radius = ball_radius_mx * avg_dist
print(f"Estimated radius for ball pivoting: {radius:.4f}")
# Reconstruct mesh using Ball Pivoting
meshRecon = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
    point_cloud, o3d.utility.DoubleVector([radius, radius*1.2, radius * 2, radius * 4, radius * 8]))

# meshRecon, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(point_cloud, depth=8)

# Compute vertex normals for the reconstructed mesh
meshRecon.compute_vertex_normals()
print(f"Ball pivoting time: {time.time() - ball_pivoting_time:.4f} seconds")

# Interpolate colors for the reconstructed mesh
recon_color_start = time.time()
recon_vertices = np.asarray(meshRecon.vertices)
recon_colors = []

for vertex in recon_vertices:
    dists = np.linalg.norm(vertex_positions - vertex, axis=1)
    nearest_idx = np.argmin(dists)
    recon_colors.append(vertex_colors[nearest_idx])

meshRecon.vertex_colors = o3d.utility.Vector3dVector(recon_colors)
print(f"Color interpolation time: {time.time() - recon_color_start:.4f} seconds")

print(f"Vertices in original mesh: {len(np.asarray(mesh.vertices))}")
print(f"Triangles in original mesh: {len(np.asarray(mesh.triangles))}")

print(f"Vertices in reconstructed mesh: {len(np.asarray(meshRecon.vertices))}")
print(f"Triangles in reconstructed mesh: {len(np.asarray(meshRecon.triangles))}")   

# meshRecon.remove_degenerate_triangles()
# meshRecon.remove_duplicated_triangles()
# meshRecon.remove_non_manifold_edges()

# # Convert legacy mesh to tensor mesh
# meshRecon_t = o3d.t.geometry.TriangleMesh.from_legacy(meshRecon)
# # Fill small holes (adjust min_hole_size as needed)
# meshRecon_t = meshRecon_t.fill_holes(hole_size=10.0)
# # Convert back to legacy mesh for visualization/export
# meshRecon = meshRecon_t.to_legacy()
#  # Try adjusting size

# =====================
# Save and Visualize Reconstruction
# =====================
o3d.io.write_triangle_mesh("mesh_edit/remesh.ply", meshRecon, write_ascii=True)
o3d.visualization.draw_geometries([meshRecon], width=1200, height=800)

# # Visualize original and remeshed mesh side by side
# # Compute bounding box and translation
# bbox = mesh.get_axis_aligned_bounding_box()
# extent = bbox.get_extent()
# gap = 0.2 * max(extent)
# translation = np.array([extent[0] + gap, 0.0, 0.0])
# # Make a copy of meshRecon to avoid modifying the original
# meshRecon_shifted = meshRecon.translate(translation, relative=False)
# # Visualize both meshes
# o3d.visualization.draw_geometries([mesh, meshRecon_shifted], window_name="Original vs Remesh Side-by-Side", width=1200, height=800)

# =====================
# Surface-to-Surface Distance Metrics
# =====================
# Sample surface points from each mesh
sample_pts_orig = sample_surface_points(mesh, 15000)
sample_pts_recon = sample_surface_points(meshRecon, 15000)

# Compute distances from orig → recon and recon → orig
dists_orig_to_recon = compute_surface_distance(sample_pts_orig, meshRecon)
dists_recon_to_orig = compute_surface_distance(sample_pts_recon, mesh)

# Combine stats
surface_to_surface_distances = np.concatenate([dists_orig_to_recon, dists_recon_to_orig])
mean_s2s = np.mean(surface_to_surface_distances)
max_s2s = np.max(surface_to_surface_distances)
median_s2s = np.median(surface_to_surface_distances)

print(f"Surface-to-Surface Mean Distance: {mean_s2s:.6f} meters")
print(f"Surface-to-Surface Median Distance: {median_s2s:.6f} meters")
print(f"Surface-to-Surface Max Distance: {max_s2s:.6f} meters")

# =====================
# F-score Evaluation
# =====================
bbox = mesh.get_axis_aligned_bounding_box()
thresh = 0.004 * np.linalg.norm(bbox.get_extent())
precision, recall, fscore = compute_fscore(sample_pts_recon, sample_pts_orig, thresh)

print(f"\nF-score (tau = {thresh:.6f} m):")
print(f"  Precision: {precision:.4f}")
print(f"  Recall:    {recall:.4f}")
print(f"  F-score:   {fscore:.4f}")

# =====================
# Chamfer and Hausdorff Distances
# =====================
print("Computing Chamfer and Hausdorff distances...")

# Sample points for distance metrics
original_pcd = mesh.sample_points_uniformly(number_of_points=10000)
reconstructed_pcd = meshRecon.sample_points_uniformly(number_of_points=10000)

# Distance from original -> recon
dist1 = original_pcd.compute_point_cloud_distance(reconstructed_pcd)
dist1 = np.asarray(dist1)

# Distance from recon -> original
dist2 = reconstructed_pcd.compute_point_cloud_distance(original_pcd)
dist2 = np.asarray(dist2)

# Chamfer distance = mean bidirectional
chamfer_dist = np.mean(dist1) + np.mean(dist2)
print(f"Chamfer Distance (avg shape difference): {chamfer_dist:.6f}")

# Hausdorff distance = max bidirectional
hausdorff_dist = max(np.max(dist1), np.max(dist2))
print(f"Hausdorff Distance (max deviation): {hausdorff_dist:.6f}")

# =====================
# Heatmap Visualization of Reconstruction Error
# =====================
# Compute per-vertex distance from reconstructed mesh to original mesh
recon_vertices = np.asarray(meshRecon.vertices)
vertex_pcd = o3d.geometry.PointCloud()
vertex_pcd.points = o3d.utility.Vector3dVector(recon_vertices)
vertex_distances = vertex_pcd.compute_point_cloud_distance(original_pcd)
vertex_distances = np.asarray(vertex_distances)

# Normalize distances and map to colors
norm_dist = vertex_distances / np.max(vertex_distances)
colors = plt.cm.viridis(norm_dist)[:, :3]
meshRecon.vertex_colors = o3d.utility.Vector3dVector(colors)

# Show colorbar using matplotlib
import matplotlib as mpl
fig, ax = plt.subplots(figsize=(6, 1))
fig.subplots_adjust(bottom=0.5)
norm = mpl.colors.Normalize(vmin=vertex_distances.min(), vmax=vertex_distances.max())
cb1 = mpl.colorbar.ColorbarBase(ax, cmap=plt.cm.viridis, norm=norm, orientation='horizontal')
cb1.set_label('Reconstruction Error (distance)')
plt.show(block=False)  # Show non-blocking
plt.pause(2)           # Pause to let it render (adjust time as needed)
plt.close(fig)         # Close the colorbar window automatically

# Visualize heatmapped mesh
print("Visualizing reconstruction error as a heatmap...")
o3d.visualization.draw_geometries([meshRecon], window_name="Reconstruction Error Heatmap")
