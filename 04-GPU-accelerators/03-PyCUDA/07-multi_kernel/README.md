## About this example

This example shows the use of multiple kernels in a single program

## Requirements

You must have Python and PIP installed in your system. PyCUDA can be installed through PIP:

```bash
$ pip install pycuda
```

If you have problems trying to install pycuda, check out [this post](https://wiki.tiker.net/PyCuda/Installation).

## Run

Open a terminal and type:

```bash
> sh run.sh
```

## Output
A typical output should look like this one. 

```
Vector Operation (A/B+C*B)
Vector Size: 1024
--------------------------------------------------------------------------------
GPU-CPU Diference [ 0.  0.  0. ...,  0.  0.  0.]
Time CPU: 0.00333905220032
Time GPU: 0.00124907493591
```

## Extra Resources

 * [OpenCL Programming Guide 1.2 Examples](https://github.com/bgaster/opencl-book-samples).
 * [NVIDIA toolkit documentation](https://developer.nvidia.com/cuda-toolkit).
