## About this example

This example shows an multiplication program of two vectors.

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
Resultado CPU: [  6.62592053e-02   6.92918718e-01  -4.74032800e-04 ...,   5.17811060e-01
   1.05464137e+00  -1.02088168e-01]
Resultado GPU: [  6.62592053e-02   6.92918718e-01  -4.74032800e-04 ...,   5.17811060e-01
   1.05464137e+00  -1.02088168e-01]
Check: 
[ 0.  0.  0. ...,  0.  0.  0.]
Vector Multiplication
Vector Size: 1024
Time CPU: 0.00119495391846
Time GPU: 0.00100517272949
```

## Extra Resources

 * [OpenCL Programming Guide 1.2 Examples](https://github.com/bgaster/opencl-book-samples).
 * [NVIDIA toolkit documentation](https://developer.nvidia.com/cuda-toolkit).
