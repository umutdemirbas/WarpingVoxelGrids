# Warping Voxel-grids for 3D Robotic Mapping

This repo is for the "Warping Voxel Grids for 3D Robotic Mapping" Project at ETH Zürich, 3D Vision class. The repo mainly focuses on mesh side of the project. For the integrated ORB-SLAM3 & Voxblox pipeline, see the submodule

## Overview

The mesh_edit directory contains the implementation of the Laplacian mesh editing algorithm as well as the mesh reconstruction process. The mesh_raycast directory comprises the source code responsible for locating corresponding points within the keyframes on the mesh.

## Installation

### Prerequisites

- Python 3.10

### Steps

```bash
git clone https://github.com/umutdemirbas/WarpingVoxelGrids.git
cd WarpingVoxelGrids
pip install -r requirements.txt
``` 

### Data

Download the data for raycasting from https://polybox.ethz.ch/index.php/s/ndRrw4RKSGrBcPF and place into the `data` folder

