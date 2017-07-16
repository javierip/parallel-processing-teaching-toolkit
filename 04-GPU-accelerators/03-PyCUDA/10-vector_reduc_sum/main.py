import time
import numpy as np
from pycuda import driver, compiler, gpuarray, tools

# -- initialize the device
import pycuda.autoinit

kernel_code_template = """
__global__ void vectorReduce(volatile float *g_idata, volatile float *g_odata)
{
    __shared__ float sdata[1024];

    int tid = threadIdx.x;
    int i = blockIdx.x * (blockDim.x ) + threadIdx.x;
    sdata[tid] = g_idata[i];

    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s ) {
            sdata[tid] += sdata[tid + s];
            __syncthreads();
        }
    }

    if (tid == 0) {
        g_odata[0] = sdata[0];

    }
}
"""

# define the (square) matrix size
#  note that we'll only use *one* block of threads here
#  as a consequence this number (squared) can't exceed max_threads,
#  see http://documen.tician.de/pycuda/util.html#pycuda.tools.DeviceData
#  for more information on how to get this number for your device
VECTOR_LEN = 1024

# create two random square matrices
a_cpu = np.random.randn(VECTOR_LEN).astype(np.float32)


# compute reference on the CPU to verify GPU computation
print "-" * 80
tic=time.time()
summ = 0
for i in range (0,VECTOR_LEN):
    summ+=a_cpu[i]

time_cpu=time.time()-tic

print 'Resultado CPU:', summ
print "Time CPU:", time_cpu

# transfer host (CPU) memory to device (GPU) memory

a_gpu = gpuarray.to_gpu(a_cpu)
c_gpu = gpuarray.empty((2), np.float32)

# get the kernel code from the template 
# by specifying the constant VECTOR_LEN

kernel_code = kernel_code_template % {
    'VECTOR_LEN': VECTOR_LEN 
}

# compile the kernel code 
mod = compiler.SourceModule(kernel_code)

# get the kernel function from the compiled module
reduction = mod.get_function("vectorReduce")

# call the kernel on the card
tic=time.time()
reduction(
    # inputs
    a_gpu,
    # output
    c_gpu,
    # (only one) block of VECTOR_LEN x VECTOR_LEN threads

    block = (VECTOR_LEN, 1,1),
)
time_gpu=time.time()-tic

print "-" * 80
print 'Resultado GPU:', c_gpu[0]
print "Time GPU:", time_gpu
np.allclose(a_cpu,  a_gpu.get())
