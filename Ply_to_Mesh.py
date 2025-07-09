import numpy as np
import open3d as o3d
from plyfile import PlyData
from tqdm import tqdm
import scipy.spatial
import cupy as cp  # Import CuPy for GPU operations

# Step 0: Progress bar wrapper function
def track_progress(func, desc, *args, **kwargs):
    with tqdm(total=1, desc=desc) as pbar:
        result = func(*args, **kwargs)
        pbar.update(1)
    return result

# GPU-accelerated Densification Function with CuPy
def densify_point_cloud_gpu(pcd, voxel_size=0.01, density_threshold=5):
    pcd = pcd.voxel_down_sample(voxel_size)
    points = cp.asarray(pcd.points)
    colors = cp.asarray(pcd.colors)  # Get the colors as a CuPy array
    voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd, voxel_size=voxel_size)
    
    new_points = []
    new_colors = []
    voxels = voxel_grid.get_voxels()

    for voxel in tqdm(voxels, desc="Densifying Point Cloud"):
        grid_index = cp.array(voxel.grid_index)
        voxel_points = points[cp.all(cp.floor(points / voxel_size) == grid_index, axis=1)]
        voxel_colors = colors[cp.all(cp.floor(points / voxel_size) == grid_index, axis=1)]
        
        if len(voxel_points) > 0 and len(voxel_points) < density_threshold:
            center = cp.mean(voxel_points, axis=0)
            for _ in range(density_threshold - len(voxel_points)):
                perturbation = cp.random.normal(scale=0.001, size=3)
                new_point = center + perturbation
                new_points.append(new_point)
                
                # Assign the color of the closest point
                new_colors.append(voxel_colors[0])

    if new_points:
        new_points = cp.vstack(new_points)
        new_colors = cp.vstack(new_colors)
        new_points = cp.asnumpy(new_points)  # Convert back to NumPy for Open3D compatibility
        new_colors = cp.asnumpy(new_colors)  # Convert back to NumPy for Open3D compatibility
        pcd.points = o3d.utility.Vector3dVector(np.vstack((np.asarray(pcd.points), new_points)))
        pcd.colors = o3d.utility.Vector3dVector(np.vstack((np.asarray(pcd.colors), new_colors)))

    return pcd

# Volume Addition Function with Progress Bar and Color Assignment
def add_volume_to_point_cloud(pcd, num_points=10000):
    hull, _ = pcd.compute_convex_hull()
    points = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors)
    kdtree = scipy.spatial.KDTree(points)
    
    with tqdm(total=1, desc="Adding Volume to Point Cloud") as pbar:
        volume_points = np.asarray(hull.sample_points_poisson_disk(num_points).points)
        _, indices = kdtree.query(volume_points)
        volume_colors = colors[indices]
        
        pcd.points = o3d.utility.Vector3dVector(np.vstack((points, volume_points)))
        pcd.colors = o3d.utility.Vector3dVector(np.vstack((colors, volume_colors)))
        
        pbar.update(1)
    return pcd


def remove_excess_points_via_hull(pcd, distance_threshold=0.01):
    # Compute the convex hull of the point cloud
    hull, _ = pcd.compute_convex_hull()
    
    # Compute distances of each point from the convex hull
    distances = hull.compute_point_cloud_distance(pcd)
    distances = np.asarray(distances)
    
    # Keep points within a certain distance from the convex hull
    selected_indices = np.where(distances < distance_threshold)[0]
    pcd_filtered = pcd.select_by_index(selected_indices)
    
    return pcd_filtered


# Step 1: Convert SH color data to RGB and load the point cloud
path = "C:/Users/raisa/gaussian-splatting_main/output/Gundam4/point_cloud/point_cloud.ply"
plydata = PlyData.read(path)
xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                np.asarray(plydata.elements[0]["y"]),
                np.asarray(plydata.elements[0]["z"])), axis=1)
features_dc = np.zeros((xyz.shape[0], 3, 1))
features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])
f_d = np.transpose(features_dc, axes=(0, 2, 1))
f_d_t = f_d[:, 0, :]

# Create point cloud object
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(xyz)
pcd.colors = o3d.utility.Vector3dVector(f_d_t)

# Save the point cloud with color information
o3d.io.write_point_cloud("cht-color.ply", pcd)

# Load the point cloud from the saved file with color information
pcd = track_progress(o3d.io.read_point_cloud, "Loading Point Cloud", "cht-color.ply")

#pcd_bpa = pcd
# Visualize the point cloud after densification and volume addition
print("Before Densification and Volume Addition:")
o3d.visualization.draw_geometries([pcd])

# Step 2: Densification and Volume Addition
pcd = densify_point_cloud_gpu(pcd, voxel_size=0.01, density_threshold=5)
pcd = add_volume_to_point_cloud(pcd, num_points=5000)

# Visualize the point cloud after densification and volume addition
print("After Densification and Volume Addition:")
o3d.visualization.draw_geometries([pcd])

# Separate downsampling and outlier removal for Poisson
pcd_poisson = pcd.voxel_down_sample(voxel_size=0.003)
cl, ind = pcd_poisson.remove_statistical_outlier(nb_neighbors=20, std_ratio=1.5)
pcd_poisson = pcd_poisson.select_by_index(ind)

# Visualize the point cloud after downsampling and outlier removal for Poisson
print("After Downsampling and Outlier Removal for Poisson:")
o3d.visualization.draw_geometries([pcd_poisson])


'''# Separate downsampling and outlier removal for BPA
pcd_bpa = pcd_bpa.voxel_down_sample(voxel_size=0.09)
cl, ind = pcd_bpa.remove_statistical_outlier(nb_neighbors=25, std_ratio=2.5)
pcd_bpa = pcd_bpa.select_by_index(ind)

# Visualize the point cloud after downsampling and outlier removal for BPA
print("After Downsampling and Outlier Removal for BPA:")
o3d.visualization.draw_geometries([pcd_bpa])'''


'''print("Re-Densifying point cloud for BPA:")
pcd_bpa = densify_point_cloud_gpu(pcd_bpa, voxel_size=0.005, density_threshold=10)
pcd_bpa = add_volume_to_point_cloud(pcd_bpa, num_points=10000)

print("After Re-Densifying point cloud for BPA:")
o3d.visualization.draw_geometries([pcd_bpa])'''

# Separate downsampling and outlier removal for BPA
pcd_bpa = pcd.voxel_down_sample(voxel_size=0.01)
cl, ind = pcd_bpa.remove_statistical_outlier(nb_neighbors=15, std_ratio=3.5)
pcd_bpa = pcd_bpa.select_by_index(ind)

# Visualize the point cloud after downsampling and outlier removal for BPA
print("After Downsampling and Outlier Removal for BPA:")
o3d.visualization.draw_geometries([pcd_bpa])



# Separate downsampling and outlier removal for Alpha Shapes
pcd_alpha = pcd.voxel_down_sample(voxel_size=0.005)
cl, ind = pcd_alpha.remove_statistical_outlier(nb_neighbors=15, std_ratio=1.5)
pcd_alpha = pcd_alpha.select_by_index(ind)

# Visualize the point cloud after downsampling and outlier removal for Alpha Shapes
print("After Downsampling and Outlier Removal for Alpha Shapes:")
o3d.visualization.draw_geometries([pcd_alpha])

# Step 3: Re-estimate normals after filtering for each method
track_progress(pcd_poisson.estimate_normals, "Estimating Normals for Poisson",
               search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.05, max_nn=30))
pcd_poisson.orient_normals_consistent_tangent_plane(k=30)

track_progress(pcd_bpa.estimate_normals, "Estimating Normals for BPA",
               search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.05, max_nn=30))
pcd_bpa.orient_normals_consistent_tangent_plane(k=30)

track_progress(pcd_alpha.estimate_normals, "Estimating Normals for Alpha Shapes",
               search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.05, max_nn=30))
pcd_alpha.orient_normals_consistent_tangent_plane(k=30)



# Step 4: Poisson Surface Reconstruction
mesh_poisson, densities = track_progress(o3d.geometry.TriangleMesh.create_from_point_cloud_poisson,
                                 "Poisson Surface Reconstruction", pcd_poisson, 
                                 depth=12, width=0, scale=1.6, linear_fit=True)

# Assign colors to Poisson mesh
kdtree_poisson = scipy.spatial.KDTree(np.asarray(pcd_poisson.points))
_, indices_poisson = kdtree_poisson.query(np.asarray(mesh_poisson.vertices))
mesh_poisson.vertex_colors = o3d.utility.Vector3dVector(np.asarray(pcd_poisson.colors)[indices_poisson])

# Visualize the Poisson mesh before refinement
print("Before Poisson Mesh Refinement:")
o3d.visualization.draw_geometries([mesh_poisson])

# Step 5: Remove low-density vertices to clean the mesh
vertices_to_remove = densities < np.quantile(densities, 0.03)
mesh_poisson.remove_vertices_by_mask(vertices_to_remove)
mesh_poisson = mesh_poisson.remove_unreferenced_vertices()

# Visualize the Poisson mesh after cleaning
print("After Poisson Mesh Cleaning:")
o3d.visualization.draw_geometries([mesh_poisson])

# Step 6: Simplify the Poisson mesh and smooth
mesh_poisson = track_progress(mesh_poisson.simplify_quadric_decimation, "Simplifying Poisson Mesh", target_number_of_triangles=75000)
mesh_poisson = track_progress(mesh_poisson.filter_smooth_taubin, "Smoothing Poisson Mesh", number_of_iterations=5)

# Visualize the Poisson mesh after simplification and smoothing
print("After Poisson Mesh Simplification and Smoothing:")
o3d.visualization.draw_geometries([mesh_poisson])

# Step 7: Ball Pivoting Algorithm (BPA) for refinement
bpa_mesh = track_progress(o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting, 
                          "BPA Mesh Generation", pcd_bpa, o3d.utility.DoubleVector([0.004, 0.006, 0.008, 0.01, 0.012, 0.015, 0.02, 0.025]))

# Assign colors to BPA mesh
kdtree_bpa = scipy.spatial.KDTree(np.asarray(pcd_bpa.points))
_, indices_bpa = kdtree_bpa.query(np.asarray(bpa_mesh.vertices))
bpa_mesh.vertex_colors = o3d.utility.Vector3dVector(np.asarray(pcd_bpa.colors)[indices_bpa])

# Visualize BPA mesh before post-processing
print("Before BPA Mesh Post-Processing:")
o3d.visualization.draw_geometries([bpa_mesh])

# Step 8: Post-process BPA mesh using available functions
bpa_mesh = track_progress(bpa_mesh.filter_smooth_taubin, "Smoothing BPA Mesh", number_of_iterations=20)
bpa_mesh = bpa_mesh.remove_degenerate_triangles()
bpa_mesh = bpa_mesh.remove_duplicated_triangles()
bpa_mesh = bpa_mesh.remove_unreferenced_vertices()
bpa_mesh = bpa_mesh.remove_non_manifold_edges()

# Visualize the BPA mesh after post-processing
print("After BPA Mesh Post-Processing:")
o3d.visualization.draw_geometries([bpa_mesh])

# Step 9: Alpha Shapes Mesh
try:
    alpha_mesh = track_progress(o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape, 
                                "Alpha Shape Meshing", pcd_alpha, alpha=0.05)
    
    # Assign colors to Alpha mesh
    kdtree_alpha = scipy.spatial.KDTree(np.asarray(pcd_alpha.points))
    _, indices_alpha = kdtree_alpha.query(np.asarray(alpha_mesh.vertices))
    alpha_mesh.vertex_colors = o3d.utility.Vector3dVector(np.asarray(pcd_alpha.colors)[indices_alpha])

except Exception as e:
    print(f"Error during Alpha Shape Meshing: {e}")
    alpha_mesh = None

if alpha_mesh is not None:
    print("Alpha Shape Mesh:")
    o3d.visualization.draw_geometries([alpha_mesh])

    # Step 10: Combine Poisson, BPA, and Alpha Meshes
    combined_mesh = bpa_mesh + alpha_mesh + mesh_poisson
    combined_mesh = combined_mesh.remove_duplicated_vertices()
    combined_mesh = track_progress(combined_mesh.simplify_quadric_decimation, "Simplifying Combined Mesh", target_number_of_triangles=75000)

    print("Combined Mesh Before Final Smoothing:")
    o3d.visualization.draw_geometries([combined_mesh])

    combined_mesh = track_progress(combined_mesh.filter_smooth_taubin, "Final Smoothing", number_of_iterations=5)

    o3d.io.write_triangle_mesh("output_final_mesh_colored.ply", combined_mesh)
    print("Final Combined Mesh:")
    o3d.visualization.draw_geometries([combined_mesh], mesh_show_back_face=True)
else:
    print("Alpha Mesh was not created due to errors.")
