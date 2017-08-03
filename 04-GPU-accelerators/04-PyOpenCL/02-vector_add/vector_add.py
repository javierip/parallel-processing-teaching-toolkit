# -*- coding: utf-8 -*-

# Parallel Processing Teaching Toolkit
# PyOpenCL - Example 02
# Vector Addition
# https://github.com/javierip/parallel-processing-teaching-toolkit

import pyopencl as cl
import numpy as np

import time   # For measure the running times

VECTOR_SIZE = 50000 # Elements of vector

# Create two random vectors a & b
a_host = np.random.rand(VECTOR_SIZE).astype(np.float32)
b_host = np.random.rand(VECTOR_SIZE).astype(np.float32)
# Create a empty vector for the result
res_host= np.zeros(VECTOR_SIZE).astype(np.float32)

# Create CL context
platform = cl.get_platforms()[0]   
device = platform.get_devices()[0] #get first gpu available

print "Running: ", platform
print "On GPU: ", device

ctx = cl.Context([device])
queue = cl.CommandQueue(ctx)


# Transfer host (CPU) memory to device (GPU) memory 
mf = cl.mem_flags
a_gpu = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=a_host)
b_gpu = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=b_host)

# Kernel code
prg = cl.Program(ctx, """
__kernel void sum(__global const float *a_gpu, __global const float *b_gpu, __global float *res_gpu) {
  int gid = get_global_id(0);
  res_gpu[gid] = a_gpu[gid] + b_gpu[gid];
}
""").build()

# Create empty gpu array for the result in GPU
res_gpu = cl.Buffer(ctx, mf.WRITE_ONLY, a_host.nbytes)


tic=time.time()

#Operation using the GPU - call the kernel on the card
prg.sum(queue, a_host.shape, None, a_gpu, b_gpu, res_gpu)

time_gpu=time.time()-tic

#Clear GPU resources
res_host = np.empty_like(a_host)
cl.enqueue_copy(queue, res_host, res_gpu)

# Check on CPU with Numpy:
print(res_host - (a_host + b_host))
print(np.linalg.norm(res_host - (a_host + b_host)))
#if 0 = good

tic=time.time()

#Operation using the CPU
for i in range(0,VECTOR_SIZE):
	res_host[i]=a_host[i]+b_host[i]

time_cpu=time.time()-tic


# Print the results
print "-" * 80
print "Vector Adition"
print "Vector Size:", VECTOR_SIZE
print "Time CPU:", time_cpu
print "Time GPU:", time_gpu

