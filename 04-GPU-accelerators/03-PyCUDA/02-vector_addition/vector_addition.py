"""
Multiplies two square matrices together using a *single* block of threads and 
global memory only. Each thread computes one element of the resulting matrix.
"""
import time
import numpy as np
from pycuda import driver, compiler, gpuarray, tools

# -- initialize the device
import pycuda.autoinit


kernel_code_template = """
__global__ void  vectorAdd(volatile float *a_gpu, volatile float *b_gpu, float *res_gpu)
{

int tid = blockIdx.x * blockDim.x + threadIdx.x;

res_gpu[tid] = a_gpu[tid] + b_gpu[tid];

}
"""

# define the (square) matrix size
#  note that we'll only use *one* block of threads here
#  as a consequence this number (squared) can't exceed max_threads,
#  see http://documen.tician.de/pycuda/util.html#pycuda.tools.DeviceData
#  for more information on how to get this number for your device
VECTOR_SIZE = 1024

# create two random square matrices
a_cpu = np.random.randn(VECTOR_SIZE).astype(np.float32)
b_cpu = np.random.randn(VECTOR_SIZE).astype(np.float32)
res_cpu= np.zeros(VECTOR_SIZE).astype(np.float32)


# compute reference on the CPU to verify GPU computation
tic=time.time()

for i in range (0,VECTOR_SIZE):
    res_cpu[i]=a_cpu[i]+b_cpu[i]


time_cpu=time.time()-tic

# transfer host (CPU) memory to device (GPU) memory

a_gpu = gpuarray.to_gpu(a_cpu)
b_gpu = gpuarray.to_gpu(b_cpu)
res_gpu = gpuarray.empty((VECTOR_SIZE), np.float32)

# get the kernel code from the template
# by specifying the constant VECTOR_SIZE

kernel_code = kernel_code_template % {
    'VECTOR_SIZE': VECTOR_SIZE
}

# compile the kernel code 
mod = compiler.SourceModule(kernel_code)

# get the kernel function from the compiled module
reduction = mod.get_function("vectorAdd")

# call the kernel on the card
tic=time.time()
reduction(
    # inputs
    a_gpu,
    b_gpu,
    # output
    res_gpu,
    # (only one) block of VECTOR_SIZE x VECTOR_SIZE threads
    #grid =

    block = (VECTOR_SIZE, 1,1),
    #grid=(65536,1,1)
)
time_gpu=time.time()-tic

print "-" * 80
print 'Resultado CPU:', res_cpu
print 'Resultado GPU:', res_gpu.get()
print 'Check: '
print res_cpu-res_gpu.get()
print "Vector Addition"
print "Vector Size:", VECTOR_SIZE
print "Time CPU:", time_cpu
print "Time GPU:", time_gpu

np.allclose(a_cpu,  a_gpu.get(), res_gpu.get())
