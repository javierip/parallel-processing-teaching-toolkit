## About this example

Multiples two square matrices together using multiple blocks and shared memory. It is based on [this PyCUDA example]](https://wiki.tiker.net/PyCuda/Examples/MatrixmulTiled).

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
[[-0.40502357 -0.39427537 -1.98279798 -0.73013443]
 [ 0.56911588  1.20871603 -1.01041949  0.13351323]
 [-1.16809094  0.92038721 -1.55715024  0.45059428]
 [-0.75707769 -0.09926169  0.45051408 -0.57348758]]
--------------------------------------------------------------------------------
Matrix B (GPU):
[[ 0.33515742 -0.61393285 -0.32319018  1.25674534]
 [-0.59892803 -0.56491643  0.58565563 -0.18363287]
 [-0.54442501 -0.2935102  -0.09756925 -0.68661547]
 [ 0.43395552  0.30709025  0.12117419 -1.92618191]]
--------------------------------------------------------------------------------
Matrix C (GPU):
[[ 0.86303484  0.82914412  0.00497672  2.331182  ]
 [ 0.07484595 -0.69465345  0.6387229   0.92987269]
 [ 0.10054933  0.79260015  1.12307584 -1.43576944]
 [-0.68842882  0.2125265   0.07309869 -0.13791469]]
--------------------------------------------------------------------------------
CPU-GPU difference:
[[  0.00000000e+00   0.00000000e+00  -1.07102096e-08   0.00000000e+00]
 [  0.00000000e+00  -5.96046448e-08   0.00000000e+00  -5.96046448e-08]
 [  2.98023224e-08   0.00000000e+00   0.00000000e+00   1.19209290e-07]
 [ -5.96046448e-08   2.98023224e-08   0.00000000e+00   2.98023224e-08]]
L2 norm: 1.66278e-07


```

## Extra Resources

The [oficial documentation](https://developer.nvidia.com/pycuda) is a great place to learn more about PyCUDA.
