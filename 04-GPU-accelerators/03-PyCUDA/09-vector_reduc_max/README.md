## About this example

This example shows the reduction of a vector to find the maximum and its location.

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
--------------------------------------------------------------------------------
Resultado CPU: 2.5797
Indice CPU: 295
Time CPU: 0.000576019287109
--------------------------------------------------------------------------------
Resultado GPU: 2.57969856262
Indice GPU: 295.0
Time GPU: 0.00107789039612
```

## Extra Resources

 * [OpenCL Programming Guide 1.2 Examples](https://github.com/bgaster/opencl-book-samples).
 * [NVIDIA toolkit documentation](https://developer.nvidia.com/cuda-toolkit).
