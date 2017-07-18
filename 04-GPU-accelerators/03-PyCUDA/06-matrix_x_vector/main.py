# -*- coding: utf-8 -*-
import numpy as np
from pycuda import driver, compiler, gpuarray, tools
import time
# -- initialize the device
import pycuda.autoinit

kernel_code_template = """
__global__ void vectorXmatrix(float *matrix, float *vector, float *result)
{
    // 2D Thread ID (assuming that only *one* block will be executed)

    int tx = threadIdx.x;

    // Pvalue is used to store the element of the matrix
    // that is computed by the thread
    float value = 0;

    // Each thread loads one row of M and one column of N, 
    //   to produce one element of P.
    for (unsigned int k = 0; k < %(MATRIX_SIZE)s; ++k) {
        value += matrix[tx * %(MATRIX_SIZE)s + k] * vector[k];
    }

    // Write the matrix to device memory;
    // each thread writes one element
    result[tx]=value;
}
"""

# define the (square) matrix size
#  note that we'll only use *one* block of threads here
#  as a consequence this number (squared) can't exceed max_threads,
#  see http://documen.tician.de/pycuda/util.html#pycuda.tools.DeviceData
#  for more information on how to get this number for your device
MATRIX_SIZE = 32  #max 32

# create two random square matrices
vector= np.random.randint(10,size=MATRIX_SIZE).astype(np.float32)
matrix= np.random.randint(10,size=(MATRIX_SIZE,MATRIX_SIZE)).astype(np.float32)
matrix_dot_vector_cpu = np.zeros(MATRIX_SIZE, np.float32)


# compute reference on the CPU to verify GPU computation
tic=time.time()
matrix_dot_vector_cpu = np.dot(matrix, vector)
time_cpu=time.time()-tic

# transfer host (CPU) memory to device (GPU) memory 
matrix_gpu = gpuarray.to_gpu(matrix)
vector_gpu = gpuarray.to_gpu(vector)

# create empty gpu array for the result (C = A * B)
matrix_dot_vector_gpu = gpuarray.empty((MATRIX_SIZE), np.float32)

# get the kernel code from the template 
# by specifying the constant MATRIX_SIZE
kernel_code = kernel_code_template % {
    'MATRIX_SIZE': MATRIX_SIZE
}

# compile the kernel code 
mod = compiler.SourceModule(kernel_code)

# get the kernel function from the compiled module
matrixmul = mod.get_function("vectorXmatrix")

tic=time.time() #time measure
# call the kernel on the card
matrixmul(
    # inputs
    matrix_gpu, vector_gpu,
    # output
    matrix_dot_vector_gpu,
    # (only one) block of MATRIX_SIZE x MATRIX_SIZE threads
    block=(MATRIX_SIZE, MATRIX_SIZE, 1), #Colums , Rows , 1 

)
time_gpu=time.time()-tic #time measure
# print the results
print "-" * 80
print "Matrix (GPU):"
print matrix_gpu.get()

print "-" * 80
print "Vector (GPU):"
print vector_gpu.get()

print "-" * 80
print "Matrix C (GPU):"
print matrix_dot_vector_gpu.get()

print "-" * 80
print "CPU-GPU difference:"
print matrix_dot_vector_cpu - matrix_dot_vector_gpu.get()

print "-" * 80
print "Time CPU:" , time_cpu
print "Time GPU:" , time_gpu

np.allclose(matrix_dot_vector_cpu, matrix_dot_vector_gpu.get())
