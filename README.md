# Parallel-Computing-Assignment3
# Overview

This repository contains parallel implementations of the K-Means clustering algorithm using three different parallel programming paradigms:

**OpenMP**: For shared-memory parallelism on multi-core CPUs.
**MPI**: For distributed-memory parallelism across multiple processes (nodes).
**CUDA**: For GPU acceleration using NVIDIA's CUDA toolkit.

The implementations demonstrate how to parallelize the core steps of K-Means:

Assignment: Assign each data point to the nearest centroid.

Update: Recalculate centroids based on the mean of assigned points.

These are designed for educational purposes and performance comparison. The dataset is small for simplicity (9 points in OpenMP/MPI, 10,000 random points in CUDA), but can be scaled easily.
Key parameters:

K: Number of clusters (default: 3).
MAX_ITER: Maximum iterations (default: 100).
N: Number of points (9 for CPU versions, 10,000 for GPU).

# Features
OpenMP Version: Uses #pragma omp parallel for for point assignment and array reductions for centroid updates. Tested with 1–16 threads.
MPI Version: Distributes points across processes using MPI_Scatterv, local assignments, MPI_Allreduce for global sums, and MPI_Bcast for centroid synchronization. Tested with 1–16 processes.
CUDA Version: Kernel-based parallelism for assignment and atomic operations for partial sums. Host-device memory transfers for updates. Tested with 128–1024 threads per block.

All versions output execution time, final cluster assignments (where applicable), and centroid positions.
Requirements
Hardware/Software

**CPU Parallelism (OpenMP/MPI)**:
Multi-core CPU.
GCC compiler (with OpenMP support: gcc -fopenmp).
OpenMPI: sudo apt-get install openmpi-bin libopenmpi-dev.

**GPU Parallelism (CUDA)**:
NVIDIA GPU with CUDA support.
CUDA Toolkit (nvcc compiler).
nvidia-smi for GPU info.

**Environment**

Tested in a Jupyter/Colab-like environment with shell magic (! commands).
Linux/Unix-based system (adapt for Windows if needed).
No external libraries beyond standard math (-lm flag).

**Installation**

Clone/Download: Get the source files (kmeans_openmp.c, kmeans_mpi.c, kmeans_cuda.cu).
Install Dependencies (run in terminal or notebook cell):Bashsudo apt-get update -qq
sudo apt-get install -y openmpi-bin libopenmpi-dev > /dev/null

# For CUDA: Ensure CUDA Toolkit is installed (e.g., via NVIDIA installer)
Verify Setup: 

mpirun --version

nvcc --version

nvidia-smi  # For GPU details

OpenMP Implementation
----------------------
Compile:

    gcc -fopenmp -O2 kmeans_openmp.c -o kmeans_openmp -lm

Run:

    ./kmeans_openmp <num_threads>

Examples:

    ./kmeans_openmp 1
    ./kmeans_openmp 2
    ./kmeans_openmp 4
    ./kmeans_openmp 8
    ./kmeans_openmp 16
    
MPI Implementation
------------------
Compile:

    mpicc kmeans_mpi.c -o kmeans_mpi -lm

Run:

    mpirun -np <processes> ./kmeans_mpi

Examples:

    mpirun -np 1 ./kmeans_mpi
    mpirun -np 2 ./kmeans_mpi
    mpirun -np 4 ./kmeans_mpi
    mpirun -np 8 ./kmeans_mpi
    
CUDA Implementation
-------------------
Compile:

    nvcc kmeans_cuda.cu -o kmeans_cuda -O2

Run:

    ./kmeans_cuda <threads_per_block>

Examples:

    ./kmeans_cuda 128
    ./kmeans_cuda 256
    ./kmeans_cuda 512
    ./kmeans_cuda 1024


    
    mpirun -np 16 ./kmeans_mpi
