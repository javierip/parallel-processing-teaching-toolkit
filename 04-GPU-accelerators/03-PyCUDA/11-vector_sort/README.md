## About this example

This example shows the reduction of a vector to find the minimum and its location.

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

```--------------------------------------------------------------------------------
Resultado CPU: -3.09072
Indice CPU: 176
Time CPU: 0.00058388710022
--------------------------------------------------------------------------------
Resultado GPU: -3.09071707726
Indice GPU: 176.0
Time GPU: 0.000946044921875

```

## Extra Resources

 * [OpenCL Programming Guide 1.2 Examples](https://github.com/bgaster/opencl-book-samples).
 * [NVIDIA toolkit documentation](https://developer.nvidia.com/cuda-toolkit).
