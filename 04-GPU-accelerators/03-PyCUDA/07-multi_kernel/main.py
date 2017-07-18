import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
from pycuda import  gpuarray
import numpy as np
import time


VECTOR_SIZE = 1024

#Host variables
a_cpu =  np.random.randint(1,10,size=VECTOR_SIZE).astype(np.float32)
b_cpu =  np.random.randint(1,10,size=VECTOR_SIZE).astype(np.float32)
c_cpu =  np.random.randint(1,10,size=VECTOR_SIZE).astype(np.float32)
d_cpu =  np.random.randint(1,10,size=VECTOR_SIZE).astype(np.float32)

res_cpu_1 =  np.zeros(VECTOR_SIZE, np.float32)
res_cpu_2 =  np.zeros(VECTOR_SIZE, np.float32)
res_cpu =  np.zeros(VECTOR_SIZE, np.float32)

# transfer host (CPU) memory to device (GPU) memory
a_gpu = gpuarray.to_gpu(a_cpu)
b_gpu = gpuarray.to_gpu(b_cpu)
c_gpu = gpuarray.to_gpu(c_cpu)
d_gpu = gpuarray.to_gpu(d_cpu)

res_gpu_1 = gpuarray.empty((VECTOR_SIZE), np.float32)
res_gpu_2 = gpuarray.empty((VECTOR_SIZE), np.float32)
res_gpu = gpuarray.empty((VECTOR_SIZE), np.float32)

#Device Source
kernel_code = SourceModule("""
    __global__ void Add(float *s, float *a, float *b)
    {
        int tid = blockIdx.x * blockDim.x + threadIdx.x;

         s[tid] = a[tid] + b[tid];
    }

    __global__ void Multi(float *m, float *a, float *b)
    {
        int tid = blockIdx.x * blockDim.x + threadIdx.x;

         m[tid] = a[tid] * b[tid];
    }

        __global__ void Div(float *d, float *a, float *b)
    {
          int tid = blockIdx.x * blockDim.x + threadIdx.x;

         d[tid] = a[tid] / b[tid];
    }
""")


# compile the kernel code

# get the kernel function from the compiled module
gpu_multi = kernel_code.get_function("Multi")
gpu_div = kernel_code.get_function("Div")
gpu_add = kernel_code.get_function("Add")

# call the kernel on the card
tic=time.time()
gpu_multi(
    # inputs
    res_gpu_1,
    a_gpu,
    # output
    b_gpu,
    # (only one) block of MATRIX_SIZE x MATRIX_SIZE threads
    block = (VECTOR_SIZE, 1, 1),
    )

gpu_div(
    # inputs
    res_gpu_2,
    c_gpu,
    # output
    d_gpu,
    # (only one) block of MATRIX_SIZE x MATRIX_SIZE threads
    block = (VECTOR_SIZE, 1, 1),
    )

gpu_add(
    # inputs
    res_gpu,
    res_gpu_1, 
    # output
    res_gpu_2, 
    # (only one) block of MATRIX_SIZE x MATRIX_SIZE threads
    block = (VECTOR_SIZE, 1, 1),
    )

time_gpu=time.time()-tic


tic=time.time()
#Operation using the cpu
for i in range(0,VECTOR_SIZE):
    res_cpu_1[i]=a_cpu[i]*b_cpu[i]

for i in range(0,VECTOR_SIZE):
    res_cpu_2[i]=c_cpu[i]/d_cpu[i]
for i in range(0,VECTOR_SIZE):
    res_cpu[i]=res_cpu_1[i]+res_cpu_2[i]

time_cpu=time.time()-tic
#Vector multiplication by constant

print "Vector Operation (A/B+C*B)"
print "Vector Size:", VECTOR_SIZE
print "-" * 80
print "GPU-CPU Diference" , res_cpu-res_gpu.get()
print "Time CPU:", time_cpu
print "Time GPU:", time_gpu
