## About this example


This example shows the reduction of a vector to find the maximum and its location.

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
Running:  <pyopencl.Platform 'NVIDIA CUDA' at 0x558310a50600>
In GPU:  <pyopencl.Device 'GeForce GTX 750' on 'NVIDIA CUDA' at 0x558310a5e120>
<pyopencl.Context at 0x558310b42e00 on <pyopencl.Device 'GeForce GTX 750' on 'NVIDIA CUDA' at 0x558310a5e120>>
<pyopencl.cffi_cl.CommandQueue object at 0x7f3eef122050>

[  5.51572382e-01   9.37093556e-01   9.80732620e-01   5.68657815e-01
   2.68912800e-02   4.16448563e-02   1.37287816e-02   7.65284002e-01
...............................
   7.50351965e-01   2.00895071e-01   4.31216322e-02   5.62348783e-01
   2.12045401e-01   8.33292425e-01   2.95735508e-01   1.18069120e-01
   4.20550287e-01   6.91085100e-01   4.39739883e-01   1.22448310e-01]
--------------------------------------------------------------------------------
Vector Reduction with Vector Size = 256
Max CPU: 0.989604
Max GPU: 0.989604
Index CPU: 19.0
Index GPU: 19.0
Time CPU: 0.00023889541626
Time GPU: 0.00466799736023
```

## Extra Resources

 * [OpenCL Programming Guide 1.2 Examples](https://github.com/bgaster/opencl-book-samples).
 * [NVIDIA toolkit documentation](https://developer.nvidia.com/cuda-toolkit).



