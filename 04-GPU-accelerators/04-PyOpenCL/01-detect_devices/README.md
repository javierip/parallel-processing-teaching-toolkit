## About this example

This example shows the OpenCL platforms and devices in your PC.

## Requirements

OPENCL and Python. 

## Run

Open a terminal and type:

```bash
sh run.sh
```


## Output

A typical output should look like this one. 

```
--------------------------------------------------
OpenCL Plataforms & Devices
--------------------------------------------------
Platform Name : NVIDIA CUDA
Platform Vendor :       NVIDIA Corporation
Platform Version :      OpenCL 1.2 CUDA 8.0.0
Platform Profile :      FULL_PROFILE
Platform Extensions :   cl_khr_global_int32_base_atomics cl_khr_global_int32_extended_atomics cl_khr_local_int32_base_atomics cl_khr_local_int32_extended_atomics cl_khr_fp64 cl_khr_byte_addressable_store cl_khr_icd cl_khr_gl_sharing cl_nv_compiler_options cl_nv_device_attribute_query cl_nv_pragma_unroll cl_nv_copy_opts cl_nv_create_buffer
    --------------------------------------------------
     Device Name:       GeForce GTX 750
     Device Type:       GPU
     Device Max Clock Speed:    1137 MHz
     Device Compute Units:      4
     Device Local Memory:       48 KB
     Device Constant Memory:    64 KB
     Device Global Memory:      1 GB
     Device Max Buffer/Image Size:      245 MB
     Device Max Work Group SIze:        1024
    --------------------------------------------------
--------------------------------------------------

```

## Extra Resources

 * [OpenCL Programming Guide 1.2 Examples](https://github.com/bgaster/opencl-book-samples).
 * [NVIDIA toolkit documentation](https://developer.nvidia.com/cuda-toolkit).



