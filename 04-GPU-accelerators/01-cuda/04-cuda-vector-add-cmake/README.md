## About this example

This example shows how to run and compile a CUDA program using CMake

## Requirements

CUDA Toolkit and proper Drivers.

## Run

Open a terminal and type:

```bash
sh run.sh
```


## Output

A typical output should look like this one. 

```
-- The C compiler identification is GNU 4.8.4
-- The CXX compiler identification is GNU 4.8.4
-- Check for working C compiler: /usr/bin/cc
-- Check for working C compiler: /usr/bin/cc -- works
-- Detecting C compiler ABI info
-- Detecting C compiler ABI info - done
-- Detecting C compile features
-- Detecting C compile features - done
-- Check for working CXX compiler: /usr/bin/c++
-- Check for working CXX compiler: /usr/bin/c++ -- works
-- Detecting CXX compiler ABI info
-- Detecting CXX compiler ABI info - done
-- Detecting CXX compile features
-- Detecting CXX compile features - done
-- Found CUDA: /usr/local/cuda-7.5 (found version "7.5") 
-- Configuring done
-- Generating done
-- Build files have been written to: /home/javier/03-code/parallel-code-examples/04-GPU-accelerators/01-cuda/04-cuda-vector-add-cmake/build
[100%] Building NVCC (Device) object CMakeFiles/application-CUDA.dir/application-CUDA_generated_vectorAdd.cu.o
Scanning dependencies of target application-CUDA
Linking CXX executable application-CUDA
[100%] Built target application-CUDA
[Vector addition of 50000 elements]
Copy input data from the host memory to the CUDA device
CUDA kernel launch with 196 blocks of 256 threads
Copy output data from the CUDA device to the host memory
Test PASSED
Done
```

## Extra Resources

The [oficial documentation](https://developer.nvidia.com/cuda-toolkit) for CUDA.
