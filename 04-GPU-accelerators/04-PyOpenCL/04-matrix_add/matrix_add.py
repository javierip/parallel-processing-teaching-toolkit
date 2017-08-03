# -*- coding: utf-8 -*-

# Parallel Processing Teaching Toolkit
# PyOpenCL - Example 04
# Matrix Addition
# https://github.com/javierip/parallel-processing-teaching-toolkit


import pyopencl as cl
import numpy as np

import time   # For measure the running times

MATRIX_SIZE = 96 # Matrix with 96*96 elements


# Create empty matrixes
a_host = np.empty([MATRIX_SIZE, MATRIX_SIZE], dtype=int)
b_host = np.empty([MATRIX_SIZE, MATRIX_SIZE], dtype=int)
# Create a empty vector for the result
res_host= np.empty([MATRIX_SIZE, MATRIX_SIZE], dtype=int)


#Load the matrixes with known numbers
x=1;
for j in range(0,MATRIX_SIZE):
	for i in range(0,MATRIX_SIZE):
		a_host[j][i]=x
		b_host[j][i]=x
		x=x+1



# Create CL context
platform = cl.get_platforms()[0]   
device = platform.get_devices()[0] #get first gpu available

print "Running: ", platform
print "In GPU: ", device

ctx = cl.Context([device])
queue = cl.CommandQueue(ctx)

print ctx
print queue

# Transfer host (CPU) memory to device (GPU) memory 
mf = cl.mem_flags
a_gpu = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=a_host)
b_gpu = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=b_host)


# Kernel code
prg = cl.Program(ctx, """
__kernel void sum(__global const int *a_gpu, __global const int *b_gpu, __global int *res_gpu) {
	

	int tx = get_global_id(0);
	int ty = get_global_id(1);

	int ssize=96; 


	int k=0;
	for(k=0;k<ssize+1;k++){

		int a_element = a_gpu[tx+ty*ssize+k];
		int b_element =  b_gpu[tx+ty*ssize+k];
		res_gpu[tx+ty*ssize+k]= a_element + b_element;

		
		a_element = a_gpu[tx+ty*2*ssize+k];
		b_element =  b_gpu[tx+ty*2*ssize+k];
		res_gpu[tx+ty*2*ssize+k]= a_element + b_element;
		
	}
	
}
""").build()


print a_host
print "-" * 80
print b_host
print "-" * 80
# Create empty gpu array for the result 
res_gpu = cl.Buffer(ctx, mf.WRITE_ONLY, a_host.nbytes)


tic=time.time()
#Operation using the GPU - call the kernel on the card
prg.sum(queue, a_host.shape, None, a_gpu, b_gpu, res_gpu)

time_gpu=time.time()-tic

#Clear GPU resources
res_host = np.empty_like(a_host)
cl.enqueue_copy(queue, res_host, res_gpu)

print res_host
print "-" * 80

# Check on CPU with Numpy:
print(res_host - (a_host + b_host))
print(np.linalg.norm(res_host - (a_host + b_host)))

tic=time.time()

#Operation using the cpu
for j in range(0,MATRIX_SIZE):
	for i in range(0,MATRIX_SIZE):
		res_host[j][i]=a_host[j][i]+b_host[j][i]

time_cpu=time.time()-tic


# Print the results
print "-" * 80
print "Time CPU:", time_cpu
print "Time GPU:", time_gpu

