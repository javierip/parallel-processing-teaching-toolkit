## About this example

This example shows an addition program of two vectors.

## Requirements

CUDA Toolkit and Drivers. 

## Run

Open a terminal and type:

```bash
sh run.sh
```


## Output

A typical output should look like this one. 

```
Device Number: 0
  Device name: GeForce GTX 750
  Memory Clock Rate (KHz): 2505000
  Memory Bus Width (bits): 128
  Peak Memory Bandwidth (GB/s): 80.160000

[Vector addition of 500 elements]
Copy input data from the host memory to the CUDA device
CUDA kernel launch with 34 blocks of 15 threads
Copy output data from the CUDA device to the host memory
Test PASSED
Time GPU: 0.022000
Time CPU: 0.003000
Resources free from CUDA Device
Done
```

## Extra Resources

 * [OpenCL Programming Guide 1.2 Examples](https://github.com/bgaster/opencl-book-samples).
 * [NVIDIA toolkit documentation](https://developer.nvidia.com/cuda-toolkit).



