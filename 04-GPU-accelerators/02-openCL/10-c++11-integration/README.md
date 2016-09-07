## About this example

This example shows how to integrate C++ 11 and OpenCL.

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
-- Performing Test COMPILER_SUPPORTS_CXX11
-- Performing Test COMPILER_SUPPORTS_CXX11 - Success
-- Performing Test COMPILER_SUPPORTS_CXX0X
-- Performing Test COMPILER_SUPPORTS_CXX0X - Success
-- Configuring done
-- Generating done
-- Build files have been written to: /home/javier/03-code/parallel-code-examples/04-GPU-accelerators/02-openCL/10-c++11-integration/build
Scanning dependencies of target application-openCL
[100%] Building CXX object CMakeFiles/application-openCL.dir/main.cpp.o
Linking CXX executable application-openCL
[100%] Built target application-openCL
./application-openCL: /usr/local/cuda-7.5/lib64/libOpenCL.so.1: no version information available (required by ./application-openCL)
./application-openCL: /usr/local/cuda-7.5/lib64/libOpenCL.so.1: no version information available (required by ./application-openCL)
Name: NVIDIA CUDA
Vendor: NVIDIA Corporation
Version: OpenCL 1.2 CUDA 7.5.30
Extensions: cl_khr_byte_addressable_store cl_khr_icd cl_khr_gl_sharing cl_nv_compiler_options cl_nv_device_attribute_query cl_nv_pragma_unroll cl_nv_copy_opts 
Using platform: NVIDIA CUDA
Device: 0
Name: GeForce GTX 480
CL_DEVICE_OPENCL_C_VERSION: OpenCL C 1.1 
CL_DEVICE_LOCAL_MEM_TYPE: 1
CL_DEVICE_DOUBLE_FP_CONFIG: 63
Device: 1
Name: Tesla C2075
CL_DEVICE_OPENCL_C_VERSION: OpenCL C 1.1 
CL_DEVICE_LOCAL_MEM_TYPE: 1
CL_DEVICE_DOUBLE_FP_CONFIG: 63
Using device: GeForce GTX 480
 result: 
1 1 2 3 4 5 6 7 8 9 

```

## Extra Resources

 * [OpenCL Programming Guide 1.2 Examples](https://github.com/bgaster/opencl-book-samples).
 * [NVIDIA toolkit documentation](https://developer.nvidia.com/cuda-toolkit).



