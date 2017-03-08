## About this example

This example introduces [PyCUDA](https://mathema.tician.de/software/pycuda/), a CUDA wrapper that let you access CUDA API from Python.
The example is based on [this PyCUDA example](https://git.tiker.net/pycuda.git/blob/HEAD:/examples/hello_gpu.py) and it performs an array multiplication.

## Requirements

You must have Python and PIP installed in your system. PyCUDA can be installed through PIP:

```bash
$ pip install pycuda
```

If you have problems trying to install pycuda, check out [this post](http://alisonrowland.com/articles/installing-pycuda-via-pip) by Alison Rowland.

## Run

Open a terminal and type:

```bash
> sh run.sh
```

It will run `hello-pycuda.py` with 400 elements in each array.


## Output

A typical output should look like this one. In this example, we've only used arrays of 10 elements.

```
python hello-pycuda.py 10
A = [-0.63441485  0.66980141  0.55359995  1.8912518  -1.74039793  1.67768097 -0.21202016 -0.0321303   1.01898217  0.81823468]
B = [ 0.48756501  0.67295897 -0.79990852 -0.13938849 -0.14993155  0.22598247 0.45394954 -0.44813037 -0.40919131  0.66403615 ]
Output = A * B = [-0.30931848  0.45074886 -0.44282931 -0.26361874  0.26094055  0.37912649 -0.09624645  0.01439857 -0.41695866  0.5433374 ]
```

## Extra Resources

The [oficial documentation](https://documen.tician.de/pycuda/) is a great place to learn more about PyCUDA.
