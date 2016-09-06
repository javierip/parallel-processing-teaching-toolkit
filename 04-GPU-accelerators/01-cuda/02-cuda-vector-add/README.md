## About this example

This example shows how to run and compile a vector addition using CUDA.

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
[Vector addition of 50000 elements]
Copy input data from the host memory to the CUDA device
CUDA kernel launch with 196 blocks of 256 threads
Copy output data from the CUDA device to the host memory
Test PASSED
Done


```

## Extra Resources

The [oficial documentation](https://developer.nvidia.com/cuda-toolkit) for CUDA.
