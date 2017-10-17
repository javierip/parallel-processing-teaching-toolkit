## About this example
This example shows the optimal selection of block size. Base on [CUDA Pro Tip: Occupancy API Simplifies Launch Configuration](https://devblogs.nvidia.com/parallelforall/cuda-pro-tip-occupancy-api-simplifies-launch-configuration).

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
Grid size is 977, array count is 1000000, min grid size is 48
Device maxThreadsPerMultiProcessor 2048
Device warpSize 32
Launched blocks of size 1024. Theoretical occupancy: 1.000000
Data is correct

```

## Extra Resources
 * [NVIDIA toolkit documentation](https://developer.nvidia.com/cuda-toolkit).



