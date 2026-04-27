---
title: 'knn-bvh: GPU-accelerated K-nearest-neighbor search using a Linear BVH'
tags:
  - CUDA
  - GPU
  - K-nearest-neighbor
  - Bounding Volume Hierarchy
  - Geophysics
  - Computational Dynamics
authors:
  - name: Chase J. Shyu
    orcid: 0000-0002-7144-1394
    affiliation: 1
affiliations:
  - name: Institute for Geophysics, Jackson School of Geosciences, The University of Texas at Austin
    index: 1
date: 27 April 2026
bibliography: paper.bib
---

# Summary

`knn-bvh` is a high-performance, GPU-accelerated library for K-nearest-neighbor (KNN) search in 2-D and 3-D point clouds. It utilizes a Linear Bounding Volume Hierarchy (LBVH) to organize points, enabling efficient spatial queries. The library is designed to be lightweight, easy to integrate into existing C++ and CUDA projects, and capable of handling large-scale datasets with millions of points and queries. Originally developed as the GPU backend for the computational geodynamics code `DynEarthSol3D`, `knn-bvh` is now available as a standalone tool for various scientific computing applications.

# Statement of Need

K-nearest-neighbor search is a fundamental operation in many scientific and engineering domains, including computer graphics, particle-based simulations, and geospatial analysis. In computational geodynamics, specifically in frameworks like `DynEarthSol3D`, KNN is essential for mesh-to-mesh interpolation and property mapping. While many CPU-based libraries (e.g., FLANN, nanoflann) provide robust KNN functionality, they often become a bottleneck in high-performance computing (HPC) workflows that are otherwise offloaded to GPUs.

Existing GPU KNN implementations often rely on brute-force approaches, which scale poorly with dataset size ($O(N \cdot M)$), or complex spatial partitioning schemes that are difficult to implement and maintain. `knn-bvh` addresses this by providing an efficient LBVH-based implementation that scales logarithmically with the number of points. Key features include:

- **Automatic GPU Memory Management**: Intelligent buffer allocation with safety margins to prevent out-of-memory errors.
- **Batched Processing**: A dedicated API for handling query sets larger than available GPU memory.
- **GPU-Direct Path**: Support for zero-copy queries where query points are already resident in device memory.
- **LBVH Efficiency**: Leverages the fast parallel BVH construction algorithm proposed by Karras (2012).

# Implementation Details

The core of `knn-bvh` is built upon a modified version of the `lbvh` library, implementing the "Maximizing Parallelism in the Construction of BVHs, Octrees, and k-d Trees" algorithm. 

### LBVH Construction
Points are first assigned Morton codes based on their spatial coordinates within the bounding box of the dataset. These codes are then sorted, and the BVH structure is built bottom-up in parallel. This approach is highly efficient on modern GPU architectures.

### Search Algorithm
The search is implemented as a CUDA kernel that performs a depth-first traversal of the BVH. To optimize performance for different values of $K$ (up to 32), the library employs a tiered strategy:
- **K-Tiers**: The kernel is specialized via templates for different ranges of $K$ (e.g., $K=1$, $K \le 20$, $K \le 32$) to optimize register usage and occupancy.
- **Two-Tier Stack**: Traversal uses a high-speed register-based stack supplemented by a global memory stack for deeper trees, ensuring both speed and robustness against stack overflow.
- **Estimated Radius Pruning**: Users can provide optional per-query search radius hints to prune traversal early, significantly accelerating queries in local search scenarios.

### Compatibility
The library provides a plain C++ header (`knn_bvh.hpp`) that avoids CUDA-specific syntax, allowing it to be integrated into standard C++ projects and compiled with standard compilers like `g++` or `nvc++` (with OpenACC/OpenMP support).

# Acknowledgements

This software was originally developed as part of the `DynEarthSol3D` project. The author acknowledges the contributions of the GeoFLAC research community. This work was supported by the Institute for Geophysics at the University of Texas at Austin.

# References

Karras, T. (2012). Maximizing parallelism in the construction of BVHs, octrees, and k-d trees. In *Proceedings of the Fourth Eurographics Conference on High-Performance Graphics* (pp. 33-37). Eurographics Association, doi:10.2312/EGGH/HPG12/033-037.
