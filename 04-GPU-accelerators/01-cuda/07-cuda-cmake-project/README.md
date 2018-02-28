## About this example

This example shows the reation of a CUDA project that contains multiple archives.

## Requirements

CUDA Toolkit and Drivers. 

## Run

Open a terminal and type:

```bash
sh run.sh
```


## Output

A typical output should look like this one. 

```
Scanning dependencies of target CUDAproject
[ 66%] Building CXX object CMakeFiles/CUDAproject.dir/lib.cpp.o
[100%] Linking CXX executable CUDAproject
[100%] Built target CUDAproject
Device Number: 0
  Device name: GeForce GTX TITAN X
  Memory Clock Rate (KHz): 3505000
  Memory Bus Width (bits): 384
  Peak Memory Bandwidth (GB/s): 336.480000

[Vector addition of 500 elements]
Copy input data from the host memory to the CUDA device
CUDA kernel launch with 34 blocks of 15 threads
Copy output data from the CUDA device to the host memory
Test PASSED
Time GPU: 0.009000
Time CPU: 0.001000
Resources free from CUDA Device
Done
```

## Extra Resources

  * [NVIDIA toolkit documentation](https://developer.nvidia.com/cuda-toolkit).
  * [A blog containing a post](http://bikulov.org/blog/2013/12/24/example-of-cmake-file-for-cuda-plus-cpp-code/)



