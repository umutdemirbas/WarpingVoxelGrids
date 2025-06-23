import open3d as o3d
from open3d import data
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import lsqr
import copy

# =====================
# Utility Functions
# =====================

def compute_vertex_adjacency(faces: np.ndarray, n_vertices: int):
    """
    Build a list of neighbor sets for each vertex.
    faces: (M×3) array of triangle indices.
    Returns: list of sets, where each set contains the indices of neighboring vertices for each vertex.
    """
    neighbors = [set() for _ in range(n_vertices)]
    for tri in faces:
        i, j, k = tri
        neighbors[i].update((j, k))
        neighbors[j].update((i, k))
        neighbors[k].update((i, j))
    return neighbors


def get_laplacian_matrix_umbrella(mesh, anchors_idx):
    """
    Build the Laplacian matrix with anchor constraints for mesh editing.
    The matrix is (n+k) x n, where n is the number of vertices and k is the number of anchors.
    The top n x n block is the uniform Laplacian (D-A),
    and the bottom k rows are identity rows for the anchors.
    """
    V = np.asarray(mesh.vertices)
    F = np.asarray(mesh.triangles)
    n = V.shape[0]
    k = len(anchors_idx)

    neighbors = compute_vertex_adjacency(F, n)

    # Assemble COO entries --> forming the matrix
    I, J, vals = [], [], []
    # Uniform Laplacian: degree on diagonal, –1 for each neighbor
    for i, nbrs in enumerate(neighbors):
        deg = len(nbrs)
        I.append(i)
        J.append(i)
        vals.append(deg)
        for j in nbrs:
            I.append(i)
            J.append(j)
            vals.append(-1)

    # Anchor identity rows
    for a, idx in enumerate(anchors_idx):
        I.append(n + a)
        J.append(idx)
        vals.append(1.0)

    # Build sparse matrix
    L = coo_matrix((vals, (I, J)), shape=(n + k, n)).tocsr()
    return L


def solve_laplacian_editing(
    mesh: o3d.geometry.TriangleMesh, anchor_ids: np.ndarray, anchor_pos: np.ndarray
) -> o3d.geometry.TriangleMesh:
    """
    Perform Laplacian mesh editing on `mesh`, pinning vertices
    at indices `anchor_ids` to the positions in `anchor_pos`.
    Solves L*V = b using least squares, where L is the Laplacian matrix with anchors.
    Returns a new mesh with updated vertex positions.
    """
    # Extract NumPy arrays
    V = np.asarray(mesh.vertices)  # (N,3)
    F = np.asarray(mesh.triangles)  # (M,3)
    N = V.shape[0]
    K = len(anchor_ids)

    # Build the Laplacian + anchors matrix L
    L = get_laplacian_matrix_umbrella(mesh, anchor_ids)

    # Build right‐hand side b = [ (D−A)V ; anchor_pos ]
    b = np.vstack(
        [(L[:N] @ V), np.zeros((K, 3))]  # delta coords: (N,3)
    )  # placeholder for anchors
    b[N:, :] = anchor_pos  # fill in the K rows

    # Solve separately per coordinate
    V_new = np.zeros_like(V)
    for dim in range(3):
        # lsqr returns (solution, istop, itn, normr) — we only want [0]
        x, *_ = lsqr(L, b[:, dim])
        V_new[:, dim] = x

    # Update and return mesh
    mesh.vertices = o3d.utility.Vector3dVector(V_new)
    mesh.compute_vertex_normals()
    return mesh

# === Helper functions for common anchor manipulations ===

def two_point_stretch(V: np.ndarray, delta: float = 0.2):
    """
    Stretch mesh by moving the two extreme X vertices apart.
    Returns (anchor_ids, anchor_pos).
    """
    idx_min_x = np.argmin(V[:, 0])
    idx_max_x = np.argmax(V[:, 0])
    anchor_ids = np.array([idx_min_x, idx_max_x])
    anchor_pos = np.vstack(
        [
            V[idx_min_x] + np.array([-delta, 0.0, 0.0]),
            V[idx_max_x] + np.array([delta, 0.0, 0.0]),
        ]
    )
    return anchor_ids, anchor_pos


def single_point_pull_with_base_fix(V, delta=0.5):
    """
    Move the north pole up and fix a base vertex to hold the mesh in place.
    Returns (anchor_ids, anchor_pos).
    """
    # 1) North pole goes up
    idx_north = np.argmax(V[:, 2])
    # 2) Fix one “base” vertex to hold the mesh in place
    idx_base = np.argmin(V[:, 2])

    anchor_ids = np.array([idx_north, idx_base])
    anchor_pos = np.vstack(
        [
            V[idx_north] + [0, 0, delta],  # move top upward
            V[idx_base],  # keep bottom where it is
        ]
    )
    return anchor_ids, anchor_pos


def three_point_motion(V: np.ndarray, delta: float = 0.1):
    """
    Move three extremal vertices to simulate a rigid-like motion.
    Returns (anchor_ids, anchor_pos).
    """
    idx_min_x = np.argmin(V[:, 0])
    idx_min_y = np.argmin(V[:, 1])
    idx_min_z = np.argmin(V[:, 2])
    anchor_ids = np.array([idx_min_x, idx_min_y, idx_min_z])
    anchor_pos = np.vstack(
        [
            V[idx_min_x] + np.array([-delta, 0.0, 0.0]),
            V[idx_min_y] + np.array([0.0, -delta, 0.0]),
            V[idx_min_z] + np.array([0.0, 0.0, -delta]),
        ]
    )
    return anchor_ids, anchor_pos


def region_lift(V: np.ndarray, threshold: float = 0.0, lift: float = 0.3):
    """
    Lift all vertices above a Z threshold and anchor the bottom ring to prevent global translation.
    Returns (anchor_ids, anchor_pos).
    """
    # select vertices above threshold to lift
    mask_lift = V[:, 2] > threshold
    lift_ids = np.where(mask_lift)[0]
    lift_pos = V[mask_lift] + np.array([0.0, 0.0, lift])

    # anchor the entire bottom ring to prevent global translation
    min_z = np.min(V[:, 2])
    bottom_ids = np.where(np.isclose(V[:, 2], min_z, atol=1e-6))[0]
    bottom_pos = V[bottom_ids]

    # combine lift and bottom anchors
    anchor_ids = np.concatenate([lift_ids, bottom_ids])
    anchor_pos = np.vstack([lift_pos, bottom_pos])

    return anchor_ids, anchor_pos


def knot_pull_demo(V: np.ndarray, lift_height: float = 1.0, base_thresh: float = -5.0):
    """
    Lift the upper part of the knot and fix a base ring to keep it in place.
    Returns (anchor_ids, anchor_pos).
    """
    # Lift vertices with high Z value
    z_thresh = np.percentile(V[:, 2], 90)  # top 10%
    lift_mask = V[:, 2] > z_thresh
    lift_ids = np.where(lift_mask)[0]
    lift_pos = V[lift_ids] + np.array([0.0, 0.0, lift_height])

    # Anchor bottom vertices
    base_mask = V[:, 2] < base_thresh
    base_ids = np.where(base_mask)[0]
    base_pos = V[base_ids]

    # Combine
    anchor_ids = np.concatenate([lift_ids, base_ids])
    anchor_pos = np.vstack([lift_pos, base_pos])
    return anchor_ids, anchor_pos


def knot_twist_demo(V: np.ndarray, angle_deg: float = 60.0, base_thresh: float = -5.0):
    """
    Twist the top part of the knot around the Z-axis while anchoring the base.
    Returns (anchor_ids, anchor_pos).
    """
    angle_rad = np.deg2rad(angle_deg)
    cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
    Rz = np.array([[cos_a, -sin_a, 0], [sin_a, cos_a, 0], [0, 0, 1]])

    # Top vertices to twist
    z_thresh = np.percentile(V[:, 2], 90)
    top_mask = V[:, 2] > z_thresh
    top_ids = np.where(top_mask)[0]
    top_pos = (Rz @ V[top_ids].T).T

    # Bottom vertices to fix
    base_mask = V[:, 2] < base_thresh
    base_ids = np.where(base_mask)[0]
    base_pos = V[base_ids]

    # Combine anchors
    anchor_ids = np.concatenate([top_ids, base_ids])
    anchor_pos = np.vstack([top_pos, base_pos])
    return anchor_ids, anchor_pos


def find_ring_vertices(mesh, start_idx, ring_distance=3):
    """
    Find all vertices that are exactly `ring_distance` steps away from `start_idx` on the mesh.
    Returns (indices, coordinates).
    """
    V = np.asarray(mesh.vertices)
    F = np.asarray(mesh.triangles)
    n = V.shape[0]
    neighbors = compute_vertex_adjacency(F, n)

    visited = set([start_idx])
    current_ring = set([start_idx])
    for _ in range(ring_distance):
        next_ring = set()
        for idx in current_ring:
            next_ring.update(neighbors[idx])
        next_ring -= visited
        visited.update(next_ring)
        current_ring = next_ring
        if not current_ring:
            break
    indices = np.array(sorted(current_ring))
    coords = V[indices]
    return indices, coords


def get_all_but_ring_anchors(mesh, center_indices, ring_distance=3):
    """
    Returns anchor indices and positions for all vertices except those within `ring_distance` of any index in `center_indices`.
    Only the 3-ring neighborhoods (and the centers) will be free to move.
    """
    V = np.asarray(mesh.vertices)
    n = V.shape[0]
    F = np.asarray(mesh.triangles)
    neighbors = compute_vertex_adjacency(F, n)
    # Accept both int and list/array input
    if isinstance(center_indices, int):
        center_indices = [center_indices]
    visited = set(center_indices)
    current_ring = set(center_indices)
    for _ in range(ring_distance):
        next_ring = set()
        for idx in current_ring:
            next_ring.update(neighbors[idx])
        next_ring -= visited
        visited.update(next_ring)
        current_ring = next_ring
        if not current_ring:
            break
    free_indices = visited  # all within <= ring_distance of any center
    all_indices = set(range(n))
    anchor_indices = np.array(sorted(list(all_indices - free_indices)))
    anchor_coords = V[anchor_indices]
    return anchor_indices, anchor_coords

# =====================
# Main Script
# =====================

if __name__ == "__main__":
    # 1) Load the mesh from file (PLY format)
    # mesh = o3d.io.read_triangle_mesh(data.KnotMesh().path)
    mesh = o3d.io.read_triangle_mesh("output_mesh_from_tsdf.ply")
    mesh.compute_vertex_normals()
    # o3d.visualization.draw_geometries([mesh], window_name="Original Mesh")

    # Copy the original mesh for comparison
    orig = copy.deepcopy(mesh)
    orig.compute_vertex_normals()

    # Visualize axes (not used in this script)
    # create a unit coordinate frame at the origin
    axis = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=100.0, origin=[0, 0, 0]  # axis length
    )

    # Get the raw vertex array
    V = np.asarray(mesh.vertices)
    N = V.shape[0]

    # This part was used in initial testing ##############
    # Find the two extreme points in X
    # idx_min_x = np.argmin(V[:, 0])
    # idx_max_x = np.argmax(V[:, 0])
    #####################################

    # Choose anchor points using helper functions
    # Uncomment and use one of the following for different editing effects:
    # anchor_ids, anchor_pos = two_point_stretch(V, 0.8)           # or single_point_pull(V), etc.
    # anchor_ids, anchor_pos = single_point_pull_with_base_fix(V, 0.6)   # does not work just translates
    # anchor_ids, anchor_pos = region_lift(V, 0.0, 0.5)
    # anchor_ids, anchor_pos = knot_pull_demo(V, lift_height=55.0, base_thresh=-5.0)
    # anchor_ids, anchor_pos = knot_twist_demo(V, angle_deg=60.0, base_thresh=-5.0)

    # === Custom anchor points for editing ===
    new_p2 = [0.375000, 5.375000, 0.536957]  # New position for vertex 10349
    edit_idx2 = 10349

    new_p = [-1.625000, 14.125000, 4.277880]  # New position for vertex 15615
    edit_idx = 15615

    # Get anchor indices and positions for all but a ring around the edit points
    anchor_ring_indices, anchor_ring_coords = get_all_but_ring_anchors(mesh, [edit_idx, edit_idx2], 50)

    # Combine the edit points and the anchor ring
    anchor_ids = np.concatenate([[edit_idx ,edit_idx2],  anchor_ring_indices])
    anchor_pos = np.vstack([new_p, new_p2, anchor_ring_coords])

    # Perform Laplacian mesh editing
    edited = solve_laplacian_editing(mesh, anchor_ids, anchor_pos)

    # Save the edited mesh to a file
    o3d.io.write_triangle_mesh("edited_mesh.ply", edited)

    # Read the saved mesh back
    loaded_mesh = o3d.io.read_triangle_mesh("edited_mesh.ply")
    loaded_mesh.compute_vertex_normals()

    # Visualize the loaded mesh
    o3d.visualization.draw_geometries(
        [loaded_mesh], window_name="Loaded Edited Mesh")

    # =====================
    # VISUALIZATION
    # =====================

    # Compute bounding box of original mesh to know how far to shift mesh2
    bbox1 = mesh.get_axis_aligned_bounding_box()
    extent = bbox1.get_extent()  # gives [size_x, size_y, size_z]

    # Shift mesh2 along X by (size_x + gap)
    gap = 0.8 * max(extent)  # e.g. 80% of the biggest dimension
    translation = np.array([extent[0] + gap, 0.0, 0.0])
    edited.translate(translation)
    edited.compute_vertex_normals()
    o3d.visualization.draw_geometries([edited], window_name="Knot edited")

    # Display original and edited mesh side by side
    o3d.visualization.draw_geometries(
        [orig, edited],
        window_name="Side-by-Side Comparison",
        width=1024,
        height=768,
        mesh_show_back_face=True,
    )
