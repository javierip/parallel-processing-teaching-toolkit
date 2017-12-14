## About this example

This program performs a matrix multiplication using the maximum amount and thread per block of the cuda device used. You can choose whether to perform the multiplication of matrix of the tilled mode or not. When the matrix is large enough to enter a block, grid is requested and it is accommodated in a grid of blocks.

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
										block 		grid
SIZE: 4 SUCCESS  - error:  2.38419e-07 (32, 32, 1) (1, 1, 1)
SIZE: 5 SUCCESS  - error:  4.76837e-07 (32, 32, 1) (1, 1, 1)
SIZE: 6 SUCCESS  - error:  2.38419e-07 (32, 32, 1) (1, 1, 1)
SIZE: 7 SUCCESS  - error:  4.76837e-07 (32, 32, 1) (1, 1, 1)
SIZE: 8 SUCCESS  - error:  4.76837e-07 (32, 32, 1) (1, 1, 1)
SIZE: 9 SUCCESS  - error:  4.76837e-07 (32, 32, 1) (1, 1, 1)

.....
```

## Extra Resources

 * [OpenCL Programming Guide 1.2 Examples](https://github.com/bgaster/opencl-book-samples).
 * [NVIDIA toolkit documentation](https://developer.nvidia.com/cuda-toolkit).
