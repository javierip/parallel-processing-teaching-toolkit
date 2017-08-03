## About this example

This example shows an vector x matrix multiplication.

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
Running:  <pyopencl.Platform 'NVIDIA CUDA' at 0x2758bd0>
On GPU:  <pyopencl.Device 'GeForce GTX TITAN X' on 'NVIDIA CUDA' at 0x274e710>
Matrix:
[[ 9.  6.  1. ...,  7.  8.  1.]
 [ 9.  0.  7. ...,  5.  2.  8.]
 [ 3.  6.  5. ...,  7.  2.  4.]
 ..., 
 [ 3.  0.  7. ...,  4.  1.  2.]
 [ 7.  5.  5. ...,  2.  8.  1.]
 [ 3.  0.  6. ...,  4.  8.  0.]]

Vector:
[ 9.  9.  8. ...,  0.  7.  1.]

Check:
[ 0.  0.  0. ...,  0.  0.  0.]
--------------------------------------------------------------------------------
Matrix x Vector : Size=  4096
Time CPU: 0.00511884689331
Time GPU: 0.0194778442383
```

## Extra Resources

 * [OpenCL Programming Guide 1.2 Examples](https://github.com/bgaster/opencl-book-samples).
 * [NVIDIA toolkit documentation](https://developer.nvidia.com/cuda-toolkit).



