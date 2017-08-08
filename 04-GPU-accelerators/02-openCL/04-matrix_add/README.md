## About this example


This example shows the addition of two matrices in CUDA.

## Requirements

CUDA Toolkit and proper Drivers.

## Run

Open a terminal and type:

```bash
sh run.sh
```


## Output

A typical output should look like this one. 

```
VECTOR A
0.0     0.0     0.0     0.0     0.0     0.0     0.0     0.0     0.0     0.0     0.0     0.00.0      0.0     0.0     0.0     0.0     0.0     0.0     0.0     0.0     0.0     0.0     0.00.0      0.0     0.0     0.0     0.0     0.0     0.0     0.0
........................................................
31.0    31.0    31.0    31.0    31.0    31.0    31.0    31.0    31.0    31.0    31.0    31.031.0    31.0    31.0    31.0    31.0    31.0    31.0    31.0    31.0    31.0    31.0    31.031.0    31.0    31.0    31.0    31.0    31.0    31.0    31.0
VECTOR B
0.0     0.0     0.0     0.0     0.0     0.0     0.0     0.0     0.0     0.0     0.0     0.00.0      0.0     0.0     0.0     0.0     0.0     0.0     0.0     0.0     0.0     0.0     0.00.0      0.0     0.0     0.0     0.0     0.0     0.0     0.0
........................................................
31.0    31.0    31.0    31.0    31.0    31.0    31.0    31.0    31.0    31.0    31.0    31.031.0    31.0    31.0    31.0    31.0    31.0    31.0    31.0    31.0    31.0    31.0    31.031.0    31.0    31.0    31.0    31.0    31.0    31.0    31.0
VECTOR R
0.0     0.0     0.0     0.0     0.0     0.0     0.0     0.0     0.0     0.0     0.0     0.00.0      0.0     0.0     0.0     0.0     0.0     0.0     0.0     0.0     0.0     0.0     0.00.0      0.0     0.0     0.0     0.0     0.0     0.0     0.0
........................................................
62.0    62.0    62.0    62.0    62.0    62.0    62.0    62.0    62.0    62.0    62.0    62.062.0    62.0    62.0    62.0    62.0    62.0    62.0    62.0    62.0    62.0    62.0    62.062.0    62.0    62.0    62.0    62.0    62.0    62.0    62.0
Vector Multiplication with 32 Elements
Tiempo GPU: 0.573000
Tiempo CPU: 0.025000
Checked operation!

Executed program succesfully.

```

## Extra Resources

 * [OpenCL Programming Guide 1.2 Examples](https://github.com/bgaster/opencl-book-samples).
 * [NVIDIA toolkit documentation](https://developer.nvidia.com/cuda-toolkit).



