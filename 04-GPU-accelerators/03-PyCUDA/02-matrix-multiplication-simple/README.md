## About this example

Multiplies two square matrices together using a *single* block of threads and global memory only. It is based on [this PyCUDA example](https://wiki.tiker.net/PyCuda/Examples/MatrixmulSimple).

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

A typical output should look like this one. In this example, we've only used arrays of 10 elements.

```
--------------------------------------------------------------------------------
Matrix A (GPU):
[[ 0.9806779  -0.81532729]
 [ 1.06310081  0.80283028]]
--------------------------------------------------------------------------------
Matrix B (GPU):
[[-1.8478657   0.38257766]
 [ 0.17386952 -2.99534345]]
--------------------------------------------------------------------------------
Matrix C (GPU):
[[-1.95392168  2.81737065]
 [-1.82487977 -1.99803376]]
--------------------------------------------------------------------------------
CPU-GPU difference:
[[  0.00000000e+00   0.00000000e+00]
 [  0.00000000e+00  -1.19209290e-07]]


```

## Extra Resources

The [oficial documentation](https://developer.nvidia.com/pycuda) is a great place to learn more about PyCUDA.
