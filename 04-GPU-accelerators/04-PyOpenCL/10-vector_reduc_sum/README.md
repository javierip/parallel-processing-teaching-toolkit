## About this example


This example shows the reduction to find the sum of a vector.

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
Running:  <pyopencl.Platform 'NVIDIA CUDA' at 0x55d92e306510>
In GPU:  <pyopencl.Device 'GeForce GTX 750' on 'NVIDIA CUDA' at 0x55d92e2d00a0>
<pyopencl.Context at 0x55d92e41cbb0 on <pyopencl.Device 'GeForce GTX 750' on 'NVIDIA CUDA' at 0x55d92e2d00a0>>
<pyopencl.cffi_cl.CommandQueue object at 0x7f5cb6658050>

[ 0.01009034  0.45293984  0.01079417  0.83157623  0.90878093  0.70061123
  0.59555459  0.79681021  0.21314494  0.0314641   0.0355786   0.08317823
...................................
  0.53474879  0.83492547  0.5472874   0.36738539  0.13216178  0.3760058
  0.9857192   0.5069688   0.57565147  0.42967793]
--------------------------------------------------------------------------------
Vector Reduction with Vector Size = 256
Result CPU: 128.382
Result GPU: 128.382
Time CPU: 0.000293970108032
Time GPU: 0.0103480815887

```

## Extra Resources

 * [OpenCL Programming Guide 1.2 Examples](https://github.com/bgaster/opencl-book-samples).
 * [NVIDIA toolkit documentation](https://developer.nvidia.com/cuda-toolkit).



