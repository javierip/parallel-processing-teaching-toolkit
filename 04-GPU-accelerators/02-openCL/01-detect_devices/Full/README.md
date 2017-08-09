## About this example

This example shows how to obtain OpenCL information.

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
Number of platforms:    1
        CL_PLATFORM_PROFILE:    FULL_PROFILE
        CL_PLATFORM_VERSION:    OpenCL 1.2 CUDA 7.5.26
        CL_PLATFORM_VENDOR:     NVIDIA Corporation
        CL_PLATFORM_EXTENSIONS: cl_khr_byte_addressable_store cl_khr_icd cl_khr_gl_sharing cl_nv_compiler_options cl_nv_device_attribute_query cl_nv_pragma_unroll cl_nv_copy_opts 
        Number of devices:      2
                CL_DEVICE_TYPE: CL_DEVICE_TYPE_GPU
                CL_DEVICE_VENDOR_ID:    4318
                CL_DEVICE_MAX_COMPUTE_UNITS:    15
                CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS:     3
        CL_DEVICE_MAX_WORK_ITEM_SIZES:  1024 1024 64 
                CL_DEVICE_MAX_WORK_GROUP_SIZE:  1024
                CL_DEVICE_PREFERRED_VECTOR_WIDTH_CHAR:  1
                CL_DEVICE_PREFERRED_VECTOR_WIDTH_SHORT: 1
                CL_DEVICE_PREFERRED_VECTOR_WIDTH_INT:   1
                CL_DEVICE_PREFERRED_VECTOR_WIDTH_LONG:  1
                CL_DEVICE_PREFERRED_VECTOR_WIDTH_FLOAT: 1
                CL_DEVICE_PREFERRED_VECTOR_WIDTH_DOUBLE:        1
                CL_DEVICE_PREFERRED_VECTOR_WIDTH_HALF:  0
                CL_DEVICE_NATIVE_VECTOR_WIDTH_CHAR:     1
                CL_DEVICE_NATIVE_VECTOR_WIDTH_SHORT:    1
                CL_DEVICE_NATIVE_VECTOR_WIDTH_INT:      1
                CL_DEVICE_NATIVE_VECTOR_WIDTH_LONG:     1
                CL_DEVICE_NATIVE_VECTOR_WIDTH_FLOAT:    1
                CL_DEVICE_NATIVE_VECTOR_WIDTH_DOUBLE:   1
                CL_DEVICE_NATIVE_VECTOR_WIDTH_HALF:     0
                CL_DEVICE_MAX_CLOCK_FREQUENCY:  1401
                CL_DEVICE_ADDRESS_BITS: 64
                CL_DEVICE_MAX_MEM_ALLOC_SIZE:   402440192
                CL_DEVICE_IMAGE_SUPPORT:        1
                CL_DEVICE_MAX_READ_IMAGE_ARGS:  128
                CL_DEVICE_MAX_WRITE_IMAGE_ARGS: 8
                CL_DEVICE_IMAGE2D_MAX_WIDTH:    16384
                CL_DEVICE_IMAGE2D_MAX_WIDTH:    16384
                CL_DEVICE_IMAGE2D_MAX_HEIGHT:   16384
                CL_DEVICE_IMAGE3D_MAX_WIDTH:    2048
                CL_DEVICE_IMAGE3D_MAX_HEIGHT:   2048
                CL_DEVICE_IMAGE3D_MAX_DEPTH:    2048
                CL_DEVICE_MAX_SAMPLERS: 16
                CL_DEVICE_MAX_PARAMETER_SIZE:   4352
                CL_DEVICE_MEM_BASE_ADDR_ALIGN:  4096
                CL_DEVICE_MIN_DATA_TYPE_ALIGN_SIZE:     128
                CL_DEVICE_SINGLE_FP_CONFIG:     CL_FP_DENORM | CL_FP_INF_NAN | CL_FP_ROUND_TO_NEAREST | CL_FP_ROUND_TO_ZERO | CL_FP_ROUND_TO_INF | CL_FP_FMA
                CL_DEVICE_SINGLE_FP_CONFIG:     CL_READ_ONLY_CACHE | CL_READ_WRITE_CACHE
                CL_DEVICE_GLOBAL_MEM_CACHE_TYPE:        CL_READ_WRITE_CACHE
                CL_DEVICE_GLOBAL_MEM_CACHELINE_SIZE:    128
                CL_DEVICE_GLOBAL_MEM_CACHE_SIZE:        245760
                CL_DEVICE_GLOBAL_MEM_SIZE:      1609760768
                CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE:     65536
                CL_DEVICE_MAX_CONSTANT_ARGS:    9
                CL_DEVICE_LOCAL_MEM_TYPE:
                CL_DEVICE_LOCAL_MEM_SIZE:       49152
                CL_DEVICE_ERROR_CORRECTION_SUPPORT:     0
                CL_DEVICE_HOST_UNIFIED_MEMORY:  0
                CL_DEVICE_PROFILING_TIMER_RESOLUTION:   1000
                CL_DEVICE_ENDIAN_LITTLE:        1
                CL_DEVICE_AVAILABLE:    1
                CL_DEVICE_COMPILER_AVAILABLE:   1
                CL_DEVICE_EXECUTION_CAPABILITIES:       CL_EXEC_KERNEL
                CL_DEVICE_QUEUE_PROPERTIES:     CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE | CL_QUEUE_PROFILING_ENABLE
                CL_DEVICE_PLATFORM:     0x2403070
        CL_DEVICE_NAME: GeForce GTX 480
        CL_DEVICE_VENDOR:       NVIDIA Corporation
        CL_DRIVER_VERSION:      352.93
        CL_DEVICE_PROFILE:      FULL_PROFILE
        CL_DEVICE_VERSION:      OpenCL 1.1 CUDA
        CL_DEVICE_OPENCL_C_VERSION:     OpenCL C 1.1 
        CL_DEVICE_EXTENSIONS:   cl_khr_byte_addressable_store cl_khr_icd cl_khr_gl_sharing cl_nv_compiler_options cl_nv_device_attribute_query cl_nv_pragma_unroll cl_nv_copy_opts  cl_khr_global_int32_base_atomics cl_khr_global_int32_extended_atomics cl_khr_local_int32_base_atomics cl_khr_local_int32_extended_atomics cl_khr_fp64 


                CL_DEVICE_TYPE: CL_DEVICE_TYPE_GPU
                CL_DEVICE_VENDOR_ID:    4318
                CL_DEVICE_MAX_COMPUTE_UNITS:    14
                CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS:     3
        CL_DEVICE_MAX_WORK_ITEM_SIZES:  1024 1024 64 
                CL_DEVICE_MAX_WORK_GROUP_SIZE:  1024
                CL_DEVICE_PREFERRED_VECTOR_WIDTH_CHAR:  1
                CL_DEVICE_PREFERRED_VECTOR_WIDTH_SHORT: 1
                CL_DEVICE_PREFERRED_VECTOR_WIDTH_INT:   1
                CL_DEVICE_PREFERRED_VECTOR_WIDTH_LONG:  1
                CL_DEVICE_PREFERRED_VECTOR_WIDTH_FLOAT: 1
                CL_DEVICE_PREFERRED_VECTOR_WIDTH_DOUBLE:        1
                CL_DEVICE_PREFERRED_VECTOR_WIDTH_HALF:  0
                CL_DEVICE_NATIVE_VECTOR_WIDTH_CHAR:     1
                CL_DEVICE_NATIVE_VECTOR_WIDTH_SHORT:    1
                CL_DEVICE_NATIVE_VECTOR_WIDTH_INT:      1
                CL_DEVICE_NATIVE_VECTOR_WIDTH_LONG:     1
                CL_DEVICE_NATIVE_VECTOR_WIDTH_FLOAT:    1
                CL_DEVICE_NATIVE_VECTOR_WIDTH_DOUBLE:   1
                CL_DEVICE_NATIVE_VECTOR_WIDTH_HALF:     0
                CL_DEVICE_MAX_CLOCK_FREQUENCY:  1147
                CL_DEVICE_ADDRESS_BITS: 64
                CL_DEVICE_MAX_MEM_ALLOC_SIZE:   1409138688
                CL_DEVICE_IMAGE_SUPPORT:        1
                CL_DEVICE_MAX_READ_IMAGE_ARGS:  128
                CL_DEVICE_MAX_WRITE_IMAGE_ARGS: 8
                CL_DEVICE_IMAGE2D_MAX_WIDTH:    16384
                CL_DEVICE_IMAGE2D_MAX_WIDTH:    16384
                CL_DEVICE_IMAGE2D_MAX_HEIGHT:   16384
                CL_DEVICE_IMAGE3D_MAX_WIDTH:    2048
                CL_DEVICE_IMAGE3D_MAX_HEIGHT:   2048
                CL_DEVICE_IMAGE3D_MAX_DEPTH:    2048
                CL_DEVICE_MAX_SAMPLERS: 16
                CL_DEVICE_MAX_PARAMETER_SIZE:   4352
                CL_DEVICE_MEM_BASE_ADDR_ALIGN:  4096
                CL_DEVICE_MIN_DATA_TYPE_ALIGN_SIZE:     128
                CL_DEVICE_SINGLE_FP_CONFIG:     CL_FP_DENORM | CL_FP_INF_NAN | CL_FP_ROUND_TO_NEAREST | CL_FP_ROUND_TO_ZERO | CL_FP_ROUND_TO_INF | CL_FP_FMA
                CL_DEVICE_SINGLE_FP_CONFIG:     CL_READ_ONLY_CACHE | CL_READ_WRITE_CACHE
                CL_DEVICE_GLOBAL_MEM_CACHE_TYPE:        CL_READ_WRITE_CACHE
                CL_DEVICE_GLOBAL_MEM_CACHELINE_SIZE:    128
                CL_DEVICE_GLOBAL_MEM_CACHE_SIZE:        229376
                CL_DEVICE_GLOBAL_MEM_SIZE:      5636554752
                CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE:     65536
                CL_DEVICE_MAX_CONSTANT_ARGS:    9
                CL_DEVICE_LOCAL_MEM_TYPE:
                CL_DEVICE_LOCAL_MEM_SIZE:       49152
                CL_DEVICE_ERROR_CORRECTION_SUPPORT:     1
                CL_DEVICE_HOST_UNIFIED_MEMORY:  0
                CL_DEVICE_PROFILING_TIMER_RESOLUTION:   1000
                CL_DEVICE_ENDIAN_LITTLE:        1
                CL_DEVICE_AVAILABLE:    1
                CL_DEVICE_COMPILER_AVAILABLE:   1
                CL_DEVICE_EXECUTION_CAPABILITIES:       CL_EXEC_KERNEL
                CL_DEVICE_QUEUE_PROPERTIES:     CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE | CL_QUEUE_PROFILING_ENABLE
                CL_DEVICE_PLATFORM:     0x2403070
        CL_DEVICE_NAME: Tesla C2075
        CL_DEVICE_VENDOR:       NVIDIA Corporation
        CL_DRIVER_VERSION:      352.93
        CL_DEVICE_PROFILE:      FULL_PROFILE
        CL_DEVICE_VERSION:      OpenCL 1.1 CUDA
        CL_DEVICE_OPENCL_C_VERSION:     OpenCL C 1.1 
        CL_DEVICE_EXTENSIONS:   cl_khr_byte_addressable_store cl_khr_icd cl_khr_gl_sharing cl_nv_compiler_options cl_nv_device_attribute_query cl_nv_pragma_unroll cl_nv_copy_opts  cl_khr_global_int32_base_atomics cl_khr_global_int32_extended_atomics cl_khr_local_int32_base_atomics cl_khr_local_int32_extended_atomics cl_khr_fp64 


```

## Extra Resources

 * [OpenCL Programming Guide 1.2 Examples](https://github.com/bgaster/opencl-book-samples).
 * [NVIDIA toolkit documentation](https://developer.nvidia.com/cuda-toolkit).