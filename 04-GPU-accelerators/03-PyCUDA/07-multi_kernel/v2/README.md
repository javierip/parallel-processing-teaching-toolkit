## About this example

This example shows the use of multiple kernels compiled separately in a single program 

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
What operation do you want to perform?
1.Addition
2.Multiplication
3.Divition
1
Vector Operation A+B
Vector Size: 1024
--------------------------------------------------------------------------------
GPU-CPU Diference [ 0.  0.  0. ...,  0.  0.  0.]
Time CPU: 0.00101208686829
Time GPU: 0.000786066055298

```

## Extra Resources

 * [OpenCL Programming Guide 1.2 Examples](https://github.com/bgaster/opencl-book-samples).
 * [NVIDIA toolkit documentation](https://developer.nvidia.com/cuda-toolkit).
