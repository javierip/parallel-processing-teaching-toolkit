## About this example

This example shows an multiplication of a matrix * vector.

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
Matrix (GPU):
[[ 2.  7.  1. ...,  3.  2.  0.]
 [ 5.  9.  5. ...,  6.  5.  1.]
 [ 3.  6.  2. ...,  9.  6.  5.]
 ..., 
 [ 4.  7.  6. ...,  0.  1.  4.]
 [ 7.  4.  3. ...,  9.  8.  9.]
 [ 4.  9.  2. ...,  8.  0.  5.]]
--------------------------------------------------------------------------------
Vector (GPU):
[ 5.  5.  9.  6.  4.  0.  9.  6.  2.  2.  1.  5.  8.  6.  5.  1.  9.  7.
  7.  5.  7.  6.  6.  8.  5.  9.  1.  9.  9.  2.  6.  4.]
--------------------------------------------------------------------------------
Matrix C (GPU):
[ 627.  823.  779.  812.  649.  789.  915.  612.  847.  826.  799.  787.
  693.  980.  838.  632.  901.  741.  866.  775.  546.  703.  830.  637.
  777.  848.  824.  751.  788.  781.  741.  722.]
--------------------------------------------------------------------------------
CPU-GPU difference:
[ 0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.
  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.]
--------------------------------------------------------------------------------
Time CPU: 6.50882720947e-05
Time GPU: 0.000993013381958
```

## Extra Resources

 * [OpenCL Programming Guide 1.2 Examples](https://github.com/bgaster/opencl-book-samples).
 * [NVIDIA toolkit documentation](https://developer.nvidia.com/cuda-toolkit).
