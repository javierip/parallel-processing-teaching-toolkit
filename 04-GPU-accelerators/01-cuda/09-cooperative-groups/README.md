## About this example

This example shows the use of cooperative groups in CUDA. Based on [CUDA Pro Tip: Optimized Filtering with Warp-Aggregated Atomics](https://devblogs.nvidia.com/cuda-pro-tip-optimized-filtering-warp-aggregated-atomics/).

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
[100%] Linking CXX executable application-CUDA
[100%] Built target application-CUDA
[Matrix Multiply Using CUDA] - Starting...
GPU Device 0: "GeForce GTX 1060 3GB" with compute capability 6.1

MatrixA(320,320), MatrixB(640,320)
Computing result using CUDA Kernel...
done
Performance= 448.65 GFlop/s, Time= 0.292 msec, Size= 131072000 Ops, WorkgroupSize= 1024 threads/block
Checking computed result for correctness: Result = PASS

NOTE: The CUDA Samples are not meant for performancemeasurements. Results may vary when GPU Boost is enabled.


```

## Extra Resources

 * [NVIDIA toolkit documentation](https://developer.nvidia.com/cuda-toolkit).



