## About this example


This example shows the use of multiple kernels in a single program.
Operation: (vector_a*vector_b) + (vector_c/vector_d)


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
Running:  <pyopencl.Platform 'NVIDIA CUDA' at 0x55aaa0ff2890>
On GPU:  <pyopencl.Device 'GeForce GTX 750' on 'NVIDIA CUDA' at 0x55aaa1127490>
--------------------------------------------------------------------------------
CHECK :
--------------------------------------------------------------------------------
[ 0.  0.  0. ...,  0.  0.  0.]
--------------------------------------------------------------------------------
Vector (a*b+c/d)
Vector Size: 50000
Time CPU: 0.174175024033
Time GPU: 0.0157821178436
```

## Extra Resources

 * [OpenCL Programming Guide 1.2 Examples](https://github.com/bgaster/opencl-book-samples).
 * [NVIDIA toolkit documentation](https://developer.nvidia.com/cuda-toolkit).



