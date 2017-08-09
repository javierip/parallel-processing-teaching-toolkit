# -*- coding: utf-8 -*-

# Parallel Processing Teaching Toolkit
# PyOpenCL - Example 07
# Multiple Kernels Execution
# https://github.com/javierip/parallel-processing-teaching-toolkit

import pyopencl as cl
import numpy as np

import time   # For measure the running times

VECTOR_SIZE = 50000  # Elements of vector

# Create four random vectors
a_host = np.random.rand(VECTOR_SIZE).astype(np.float32)
b_host = np.random.rand(VECTOR_SIZE).astype(np.float32)
c_host = np.random.rand(VECTOR_SIZE).astype(np.float32)
d_host = np.random.rand(VECTOR_SIZE).astype(np.float32)

# Create empty vectors for results
res_host_1= np.zeros(VECTOR_SIZE).astype(np.float32)
res_host_2= np.zeros(VECTOR_SIZE).astype(np.float32)
res_host= np.zeros(VECTOR_SIZE).astype(np.float32)
res_gpu_host= np.zeros(VECTOR_SIZE).astype(np.float32)

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
c_gpu = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=c_host)
d_gpu = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=d_host)

# Kernel code
prg = cl.Program(ctx, """
__kernel void sum(__global const float *a_gpu, __global const float *b_gpu, __global float *res_gpu) {
  int gid = get_global_id(0);
  res_gpu[gid] = a_gpu[gid] + b_gpu[gid];
}
__kernel void multi(__global const float *a_gpu, __global const float *b_gpu, __global float *res_gpu) {
  int gid = get_global_id(0);
  res_gpu[gid] = a_gpu[gid] * b_gpu[gid];
}
__kernel void div(__global const float *a_gpu, __global const float *b_gpu, __global float *res_gpu) {
  int gid = get_global_id(0);
  res_gpu[gid] = a_gpu[gid] / b_gpu[gid];
}
""").build()

# Create empty gpu array for the result 
res_gpu_1 = cl.Buffer(ctx, mf.WRITE_ONLY, a_host.nbytes)
res_gpu_2 = cl.Buffer(ctx, mf.WRITE_ONLY, a_host.nbytes)
res_gpu = cl.Buffer(ctx, mf.WRITE_ONLY, a_host.nbytes)


tic=time.time()
#Operation using the GPU - call the kernel on the card
prg.multi(queue, a_host.shape, None, a_gpu, b_gpu, res_gpu_1)
prg.div(queue, a_host.shape, None, c_gpu, d_gpu, res_gpu_2)
prg.sum(queue, a_host.shape, None, res_gpu_1, res_gpu_2, res_gpu)
time_gpu=time.time()-tic

#Clear GPU resources
res_gpu_host = np.empty_like(a_host)
cl.enqueue_copy(queue, res_gpu_host, res_gpu)



tic=time.time()

#Operation using the cpu
for i in range(0,VECTOR_SIZE):
	res_host_1[i]=a_host[i]*b_host[i]
for i in range(0,VECTOR_SIZE):
	res_host_2[i]=c_host[i]/d_host[i]
for i in range(0,VECTOR_SIZE):
	res_host[i]=res_host_1[i]+res_host_2[i]

time_cpu=time.time()-tic



# Print the results
print "-" * 80
print "CHECK :" #0 = GOOD
print "-" * 80
print res_gpu_host-res_host
print "-" * 80
print "Vector (a*b+c/d)"
print "Vector Size:", VECTOR_SIZE
print "Time CPU:", time_cpu
print "Time GPU:", time_gpu

