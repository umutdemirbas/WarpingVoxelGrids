#include <voxblox/io/layer_io.h>
#include <voxblox/core/layer.h>
#include <voxblox/mesh/mesh_layer.h>
#include <voxblox/mesh/mesh_integrator.h>
#include <voxblox/io/mesh_ply.h>
#include <iostream>

int main(int argc, char** argv) {
  using namespace voxblox;

  // Load TSDF layer from file
  Layer<TsdfVoxel>::Ptr tsdf_layer;
  std::string file_path = "/usr/home/ws/src/voxblox_docs/test_thoth_05-15.vxblx";

  // Attempt to load the TSDF layer
  if (!io::LoadLayer<TsdfVoxel>(file_path, &tsdf_layer)) {
    std::cerr << "Failed to load TSDF layer from " << file_path << std::endl;
    return -1;
  }

  std::cout << "Successfully loaded TSDF layer with "
            << tsdf_layer->getNumberOfAllocatedBlocks() << " blocks." << std::endl;

  // Create mesh layer and mesh integrator for mesh extraction
  MeshLayer mesh_layer(tsdf_layer->block_size());

  MeshIntegratorConfig config;
  config.min_weight = 1e-4f;  // Minimum weight for surface extraction
  MeshIntegrator<TsdfVoxel> mesh_integrator(config, tsdf_layer.get(), &mesh_layer);

  // Generate mesh from the TSDF layer
  mesh_integrator.generateMesh(/*only_updated_blocks=*/false, /*clear_flags=*/false);

  // Save the generated mesh to a PLY file
  std::string mesh_output_path = "/usr/home/ws/src/voxblox_docs/output_mesh_from_tsdf.ply";
  if (outputMeshLayerAsPly(mesh_output_path, mesh_layer)) {
    std::cout << "Saved mesh to " << mesh_output_path << std::endl;
  } else {
    std::cerr << "Failed to save mesh." << std::endl;
    return -1;
  }

  // Print mesh block information (block indices, vertex and face counts, and per-vertex info)
  BlockIndexList allocated_meshes;
  mesh_layer.getAllAllocatedMeshes(&allocated_meshes);

  for (const BlockIndex& index : allocated_meshes) {
    const Mesh& mesh = mesh_layer.getMeshByIndex(index);
    std::cout << "Block index: " << index.transpose()
              << ", Vertices: " << mesh.vertices.size()
              << ", Faces: " << mesh.indices.size() / 3 << std::endl;

    for (size_t i = 0; i < mesh.vertices.size(); ++i) {
      const Point& vertex = mesh.vertices[i];
      const Color& color = mesh.colors[i];
      std::cout << "  Vertex " << i << ": [" << vertex.transpose()
                << "], Color: [" << static_cast<int>(color.r) << ", "
                << static_cast<int>(color.g) << ", "
                << static_cast<int>(color.b) << "]" << std::endl;
    }
  }

  // Print TSDF layer properties (voxel size, voxels per side, block size, block volume)
  std::cout << "TSDF Layer Info:" << std::endl;
  std::cout << "  Voxel size: " << tsdf_layer->voxel_size() << " m" << std::endl;
  std::cout << "  Voxels per side: " << tsdf_layer->voxels_per_side() << std::endl;
  std::cout << "  Block size: " << tsdf_layer->block_size() << " m" << std::endl;
  std::cout << "  Block volume: " << std::pow(tsdf_layer->block_size(), 3) << " m³" << std::endl;

  return 0;
}
