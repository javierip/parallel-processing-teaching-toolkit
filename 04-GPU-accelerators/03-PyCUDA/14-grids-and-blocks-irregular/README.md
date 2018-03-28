## About this example

This program saves the thread identification values as a number in the matrices.

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

SIZE = N : MATRIX = N * N
error = max difference gpu vs cpu result


```
Blocks:  (2, 1, 1)
Grid:  (4, 8, 1)
id_blocks_x_cpu
[[ 0.  0.  1.  1.  2.  2.  3.  3.]
 [ 0.  0.  1.  1.  2.  2.  3.  3.]
 [ 0.  0.  1.  1.  2.  2.  3.  3.]
 [ 0.  0.  1.  1.  2.  2.  3.  3.]
 [ 0.  0.  1.  1.  2.  2.  3.  3.]
 [ 0.  0.  1.  1.  2.  2.  3.  3.]
 [ 0.  0.  1.  1.  2.  2.  3.  3.]
 [ 0.  0.  1.  1.  2.  2.  3.  3.]]
id_blocks_y_cpu
[[ 0.  0.  0.  0.  0.  0.  0.  0.]
 [ 1.  1.  1.  1.  1.  1.  1.  1.]
 [ 2.  2.  2.  2.  2.  2.  2.  2.]
 [ 3.  3.  3.  3.  3.  3.  3.  3.]
 [ 4.  4.  4.  4.  4.  4.  4.  4.]
 [ 5.  5.  5.  5.  5.  5.  5.  5.]
 [ 6.  6.  6.  6.  6.  6.  6.  6.]
 [ 7.  7.  7.  7.  7.  7.  7.  7.]]
id_threads_x_cpu
[[ 0.  1.  0.  1.  0.  1.  0.  1.]
 [ 0.  1.  0.  1.  0.  1.  0.  1.]
 [ 0.  1.  0.  1.  0.  1.  0.  1.]
 [ 0.  1.  0.  1.  0.  1.  0.  1.]
 [ 0.  1.  0.  1.  0.  1.  0.  1.]
 [ 0.  1.  0.  1.  0.  1.  0.  1.]
 [ 0.  1.  0.  1.  0.  1.  0.  1.]
 [ 0.  1.  0.  1.  0.  1.  0.  1.]]
id_threads_y_cpu
[[ 0.  0.  0.  0.  0.  0.  0.  0.]
 [ 0.  0.  0.  0.  0.  0.  0.  0.]
 [ 0.  0.  0.  0.  0.  0.  0.  0.]
 [ 0.  0.  0.  0.  0.  0.  0.  0.]
 [ 0.  0.  0.  0.  0.  0.  0.  0.]
 [ 0.  0.  0.  0.  0.  0.  0.  0.]
 [ 0.  0.  0.  0.  0.  0.  0.  0.]
 [ 0.  0.  0.  0.  0.  0.  0.  0.]]
id_cell_cpu
[[  0.   1.   2.   3.   4.   5.   6.   7.]
 [  8.   9.  10.  11.  12.  13.  14.  15.]
 [ 16.  17.  18.  19.  20.  21.  22.  23.]
 [ 24.  25.  26.  27.  28.  29.  30.  31.]
 [ 32.  33.  34.  35.  36.  37.  38.  39.]
 [ 40.  41.  42.  43.  44.  45.  46.  47.]
 [ 48.  49.  50.  51.  52.  53.  54.  55.]
 [ 56.  57.  58.  59.  60.  61.  62.  63.]]
...
```

## Extra Resources

 * [PyCuda Examples](https://andreask.cs.illinois.edu/PyCuda/Examples).
 * [NVIDIA toolkit documentation](https://developer.nvidia.com/cuda-toolkit).
