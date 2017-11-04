## About this example

This example shows an multiplication of two matrices.

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
Matrix A (GPU):
[[ 0.13612084  0.07987367 -1.63398063 ..., -0.61454916 -1.90155137
  -1.20777559]
 ..., 
 [-0.61635107  0.39776012  2.00158024 ...,  1.45984018  0.22637476
   0.30090544]]
--------------------------------------------------------------------------------
Matrix B (GPU):
[[ 1.48104918  1.03540909 -0.67718965 ..., -0.35710877 -0.95803785
   1.32211721]
    ..., 
 [-0.71108127  0.75352812 -1.7662549  ...,  1.28564429 -0.24351227
  -0.28769019]]
--------------------------------------------------------------------------------
Matrix C (GPU):
[[  2.49411583   0.18922229   1.99307001 ...,  -0.68889982   6.54136705
    9.55049419]
 ..., 
 [ -8.02693558  -9.72163582  -1.65157223 ...,  10.40832806   3.52716756
    8.01306534]]
--------------------------------------------------------------------------------
CPU-GPU difference:
[[  4.76837158e-07  -2.53319740e-07   2.38418579e-07 ...,  -5.96046448e-08
   -4.76837158e-07   0.00000000e+00]
 ..., 
 [  0.00000000e+00   0.00000000e+00   0.00000000e+00 ...,   0.00000000e+00
   -2.38418579e-07   0.00000000e+00]]
--------------------------------------------------------------------------------
Time CPU: 0.000109910964966
Time GPU: 0.000978946685791
```

## Extra Resources

 * [OpenCL Programming Guide 1.2 Examples](https://github.com/bgaster/opencl-book-samples).
 * [NVIDIA toolkit documentation](https://developer.nvidia.com/cuda-toolkit).
