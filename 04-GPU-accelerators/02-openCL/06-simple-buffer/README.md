## About this example

This example shows how to use Buffers and OpenCL.

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
-- Found OpenCL: /usr/lib/x86_64-linux-gnu/libOpenCL.so  
-- Configuring done
-- Generating done
-- Build files have been written to: /home/javier/03-code/parallel-code-examples/04-GPU-accelerators/02-openCL/06-simple-buffer/build
Scanning dependencies of target application-openCL
[100%] Building CXX object CMakeFiles/application-openCL.dir/main.cpp.o
In file included from /home/javier/03-code/parallel-code-examples/04-GPU-accelerators/02-openCL/06-simple-buffer/main.cpp:22:0:
/home/javier/03-code/parallel-code-examples/04-GPU-accelerators/02-openCL/06-simple-buffer/info.hpp:395:8: warning: extra tokens at end of #endif directive [enabled by default]
 #endif __INFO_HDR__
        ^
Linking CXX executable application-openCL
[100%] Built target application-openCL
./application-openCL: /usr/local/cuda-7.5/lib64/libOpenCL.so.1: no version information available (required by ./application-openCL)
./application-openCL: /usr/local/cuda-7.5/lib64/libOpenCL.so.1: no version information available (required by ./application-openCL)
Simple buffer and sub-buffer Example
Number of platforms:    1
        CL_PLATFORM_VENDOR:     NVIDIA Corporation
                CL_DEVICE_TYPE: CL_DEVICE_TYPE_GPU
                CL_DEVICE_TYPE: CL_DEVICE_TYPE_GPU
 0 1 4 9 16 25 36 49 64 81 100 121 144 169 196 225
 256 289 324 361 400 441 484 529 576 625 676 729 784 841 900 961
Program completed successfully


```

## Extra Resources

 * [OpenCL Programming Guide 1.2 Examples](https://github.com/bgaster/opencl-book-samples).
 * [NVIDIA toolkit documentation](https://developer.nvidia.com/cuda-toolkit).



