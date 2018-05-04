# gpuMemoryTracking
Concurrency Final Project 2018 to track memory references of CUDA kernels

# Limitations
This project relies on SASSI, found [here](https://github.com/NVlabs/SASSI). As of writing this, SASSI is dependent on CUDA 7, which may have compatibility with new Linux distributions such as Xenial. 

# About
The goal of this project is to gather data on memory accesses of GPU kernels in order to later evaluate patterns to aid in the development of intelligent pre-fetching algorithms for GPU devices. We've written instrumentation code that is inserted by SASSI before each memory read or write in the lower level assembly files of targeted CUDA kernels.  

# Instrumentation
The instrumentation code can be found under [sassi_instrumentor](./sassi_instrumentor). This includes a naive and coalescing version of the instrumentor. The naive version will record all base memory addresses of each thread without regard to repeats within a warp. The coalesing version will record only the unique base memory addresses accessed by a warp for each targeted instruction.  

# Sample kernels
Small example kernels that run with the given instrumentation code can be found under [sassi_kernels](./sassi_kernels). These can be compiled using the Makefile and run as normal executables. 
