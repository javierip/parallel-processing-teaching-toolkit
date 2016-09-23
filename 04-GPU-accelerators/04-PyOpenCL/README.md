## About this example

This example introduces [PyOpenCL](https://mathema.tician.de/software/pyopencl/), an OpenCL wrapper that let you access the OpenCL API from Python.
The example is based on [this basic demo](https://github.com/pyopencl/pyopencl/blob/master/examples/demo.py) and it performs an array addition.

## Requirements

You must have Python and PIP installed in your system.
Follow [the official documentation](https://wiki.tiker.net/PyOpenCL/Installation) to install PyOpenCL according to your OS.

## Run

Open a terminal and type:

```bash
> sh run.sh
```

## Output

A typical output should look like this one. In this example, we've only used arrays of 10 elements.

```
python hello-pyopencl.py
Choose platform:
[0] <pyopencl.Platform 'NVIDIA CUDA' at 0x1ad4aa0>
Choice [0]:0
Choose device(s):
[0] <pyopencl.Device 'GeForce GTX 480' on 'NVIDIA CUDA' at 0x18000c0>
[1] <pyopencl.Device 'Tesla C2075' on 'NVIDIA CUDA' at 0x19e90b0>
Choice, comma-separated [0]:1
Set the environment variable PYOPENCL_CTX='0:1' to avoid being asked again.
[ 0.  0.  0. ...,  0.  0.  0.]
0.0
```

## Extra Resources

The [oficial documentation](https://documen.tician.de/pyopencl/) is a great place to learn more about PyOpenCL.
