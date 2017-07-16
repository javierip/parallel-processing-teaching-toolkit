## About this example

This example shows an addition program of two vectors.

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
Resultado CPU: [-0.48029369  1.9872483   2.32125473 ...,  3.13818264 -2.15658116
  0.7263515 ]
Resultado GPU: [-0.48029369  1.9872483   2.32125473 ...,  3.13818264 -2.15658116
  0.7263515 ]
Check: 
[ 0.  0.  0. ...,  0.  0.  0.]
Vector Addition
Vector Size: 1024
Time CPU: 0.00113201141357
Time GPU: 0.000977039337158
```

## Extra Resources

 * [OpenCL Programming Guide 1.2 Examples](https://github.com/bgaster/opencl-book-samples).
 * [NVIDIA toolkit documentation](https://developer.nvidia.com/cuda-toolkit).
