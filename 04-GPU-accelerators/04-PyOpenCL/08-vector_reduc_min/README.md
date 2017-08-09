## About this example

This example shows the reduction of a vector to find the minimum and its location.

## Requirements

OPENCL and Python. 

## Run

Open a terminal and type:

```bash
sh run.sh
```


## Output
A typical output should look like this one. 

```
Running:  <pyopencl.Platform 'NVIDIA CUDA' at 0x55b8ddafc3e0>
In GPU:  <pyopencl.Device 'GeForce GTX 750' on 'NVIDIA CUDA' at 0x55b8dd5ac6c0>

[ 0.75051183  0.84658062  0.3047761   0.94087136  0.18246287  0.59422565
  0.58162844  0.13727482  0.63731229  0.24569485  0.07537535  0.51686388
................................................
  0.15747365  0.68173796  0.86980772  0.62664843  0.22408016  0.78685063
  0.70175791  0.23737869  0.82440794  0.11068825  0.30794129  0.23226036
  0.25885773  0.31722498  0.93376273  0.30353844]
--------------------------------------------------------------------------------
Vector Reduction with Vector Size = 256
Min CPU: 0.00811757
Min GPU: 0.00811757
Index CPU: 164.0
Index GPU: 164.0
Time CPU: 0.000254154205322
Time GPU: 0.00465106964111
```

## Extra Resources

 * [OpenCL Programming Guide 1.2 Examples](https://github.com/bgaster/opencl-book-samples).
 * [NVIDIA toolkit documentation](https://developer.nvidia.com/cuda-toolkit).



