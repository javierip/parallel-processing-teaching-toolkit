## About this example

This example shows an additon program of two matrices.

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
Running:  <pyopencl.Platform 'NVIDIA CUDA' at 0x55921f351a10>
In GPU:  <pyopencl.Device 'GeForce GTX 750' on 'NVIDIA CUDA' at 0x55921f4fb170>
<pyopencl.Context at 0x55921f408c50 on <pyopencl.Device 'GeForce GTX 750' on 'NVIDIA CUDA' at 0x55921f4fb170>>
<pyopencl.cffi_cl.CommandQueue object at 0x7f234f8de0d0>
[[   1    2    3 ...,   94   95   96]
 [  97   98   99 ...,  190  191  192]
 [ 193  194  195 ...,  286  287  288]
 ..., 
 [8929 8930 8931 ..., 9022 9023 9024]
 [9025 9026 9027 ..., 9118 9119 9120]
 [9121 9122 9123 ..., 9214 9215 9216]]
--------------------------------------------------------------------------------
[[   1    2    3 ...,   94   95   96]
 [  97   98   99 ...,  190  191  192]
 [ 193  194  195 ...,  286  287  288]
 ..., 
 [8929 8930 8931 ..., 9022 9023 9024]
 [9025 9026 9027 ..., 9118 9119 9120]
 [9121 9122 9123 ..., 9214 9215 9216]]
--------------------------------------------------------------------------------
[[    2     4     6 ...,   188   190   192]
 [  194   196   198 ...,   380   382   384]
 [  386   388   390 ...,   572   574   576]
 ..., 
 [17858 17860 17862 ..., 18044 18046 18048]
 [18050 18052 18054 ..., 18236 18238 18240]
 [18242 18244 18246 ..., 18428 18430 18432]]
--------------------------------------------------------------------------------
GPU-CPU Difference
[[0 0 0 ..., 0 0 0]
 [0 0 0 ..., 0 0 0]
 [0 0 0 ..., 0 0 0]
 ..., 
 [0 0 0 ..., 0 0 0]
 [0 0 0 ..., 0 0 0]
 [0 0 0 ..., 0 0 0]]
0.0
--------------------------------------------------------------------------------
Time CPU: 0.0223619937897
Time GPU: 0.00595188140869
```

## Extra Resources

 * [OpenCL Programming Guide 1.2 Examples](https://github.com/bgaster/opencl-book-samples).
 * [NVIDIA toolkit documentation](https://developer.nvidia.com/cuda-toolkit).



