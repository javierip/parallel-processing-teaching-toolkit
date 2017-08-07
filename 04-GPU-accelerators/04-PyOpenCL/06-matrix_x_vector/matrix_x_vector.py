# -*- coding: utf-8 -*-

# Parallel Processing Teaching Toolkit
# PyOpenCL - Example 06
# Matrix x Vector
# https://github.com/javierip/parallel-processing-teaching-toolkit


import pyopencl as cl
from pyopencl import array
import numpy as np

import time   # For measure the running times

if __name__ == "__main__":

	VECTOR_SIZE=4096 # Elements in the vector and matrix elements = VECTOR_SIZE*VECTOR_SIZE

	# Create an random vector
	vector= np.random.randint(10,size=VECTOR_SIZE).astype(np.float32)
	# Create an random matrix
	matrix= np.random.randint(10,size=(VECTOR_SIZE,VECTOR_SIZE)).astype(np.float32)
	# Create empty vectors for the result
	matrix_dot_vector_cpu = np.zeros(VECTOR_SIZE, np.float32)
	matrix_dot_vector = np.zeros(VECTOR_SIZE, np.float32)
	
	tic=time.time()
	#Operation using the cpu
	matrix_dot_vector_cpu=np.dot(matrix,vector)

	time_cpu=time.time()-tic


	# Create CL context
	platform = cl.get_platforms()[0]   
	device = platform.get_devices()[0] #get first gpu available

	print "Running: ", platform
	print "On GPU: ", device

	ctx = cl.Context([device])
	queue = cl.CommandQueue(ctx)

	# Kernel code
	program = cl.Program(ctx, """
			__kernel void matrix_dot_vector(__global const float *matrix,
		 __global const float *vector, __global float *result)
		 {	
		 	int tx = get_global_id(0);
		 	int size=4096;
		    float value=0 ;
		   
    			for (unsigned int k = 0; k < size; ++k) {
        			value += matrix[tx * size + k] * vector[k];
   				 }
   				 result[tx]=value;
		 }
		 """).build()

	#Create GPU resources
	queue = cl.CommandQueue(ctx)
	 
	mem_flags = cl.mem_flags
	matrix_gpu = cl.Buffer(ctx, mem_flags.READ_ONLY |   
				  mem_flags.COPY_HOST_PTR, hostbuf=matrix)
	vector_gpu = cl.Buffer(ctx, mem_flags.READ_ONLY |  
				  mem_flags.COPY_HOST_PTR, hostbuf=vector)


	matrix_dot_vector_gpu = cl.Buffer(ctx, mem_flags.WRITE_ONLY, 
					 matrix_dot_vector.nbytes)

	tic=time.time()
	#Operation using the GPU - call the kernel on the card
	program.matrix_dot_vector(queue, matrix_dot_vector.shape, None,   
							   matrix_gpu, vector_gpu, matrix_dot_vector_gpu)
	time_gpu=time.time()-tic

	#Clear GPU resources
	cl.enqueue_copy(queue, matrix_dot_vector, matrix_dot_vector_gpu)
	
	print "Matrix:"
	print matrix 
	print
	print "Vector:"
	print vector 
	print
	print "Check:" 
	print matrix_dot_vector_cpu-matrix_dot_vector #if = 0 then GOOD
	print "-" * 80
	print "Matrix x Vector : Size= ", VECTOR_SIZE
	print "Time CPU:", time_cpu
	print "Time GPU:", time_gpu