import time
import numpy as np
from pycuda import driver, compiler, gpuarray, tools

# -- initialize the device
import pycuda.autoinit

kernel_code_template = """
__global__ void vectorSort(volatile float *global_input_data)
{
    
    __shared__ float sdata[%(VECTOR_LEN)s];
    
    int tid = threadIdx.x;
    int i = blockIdx.x * (blockDim.x ) + threadIdx.x;
    sdata[tid] = global_input_data[i];
    
    __syncthreads();
    
    int position;
    float swap;
    
    int c=tid;
    position = c;
    
    for (unsigned int d = c + 1 ; d < %(VECTOR_LEN)s  ; d++ )
        if ( sdata[position] > sdata[d] )
            position = d;
    
    __syncthreads();
    
    if ( position != c )
    {
        swap = sdata[c];
        sdata[c] = sdata[position];
        sdata[position] = swap;
    }
    
    
    global_input_data[tid]=sdata[tid];
}
"""


# excecute sort in CPU
def cpu_sort(vector_cpu, VECTOR_LEN):
    for c in range(0, VECTOR_LEN - 1):

        position = c;

        for d in range(c + 1, VECTOR_LEN):

            if (vector_cpu[position] > vector_cpu[d]):
                position = d;

        if (position != c):
            swap = vector_cpu[c];
            vector_cpu[c] = vector_cpu[position];
            vector_cpu[position] = swap;

    return vector_cpu


# execute sort in GPU
def gpu_sort(vector_gpu, sort_binary_gpu, VECTOR_LEN):
    sort_binary_gpu(
        # inputs
        vector_gpu,

        # (only one) block of VECTOR_LEN x VECTOR_LEN threads
        block=(VECTOR_LEN, 1, 1),
    )
    return vector_gpu


def compare_sort_operations(length):
    VECTOR_LEN = length

    # create a vector of random float numbers
    vector_cpu = np.random.randn(VECTOR_LEN).astype(np.float32)

    print vector_cpu
    print "-" * 80
    # get the kernel code from the template
    # by specifying the constant VECTOR_LEN
    kernel_code = kernel_code_template % {'VECTOR_LEN': VECTOR_LEN}

    # compile the kernel code
    mod = compiler.SourceModule(kernel_code)

    # get the kernel function from the compiled module
    sort_binary_gpu = mod.get_function("vectorSort")

    # CPU sort
    tic = time.time()
    vector_cpu = cpu_sort(vector_cpu, VECTOR_LEN)
    time_cpu = time.time() - tic

    print "-" * 80

    # GPU sort
    tic = time.time()
    vector_gpu = gpuarray.to_gpu(vector_cpu)
    vector_gpu = gpu_sort(vector_gpu, sort_binary_gpu, VECTOR_LEN)
    time_gpu = time.time() - tic

    print 'Result CPU:'
    print vector_cpu
    print "-" * 80
    print 'Result GPU:'
    print vector_gpu
    print "-" * 80

    print "Time GPU:", time_gpu
    print "Time CPU:", time_cpu


if __name__ == "__main__":
    compare_sort_operations(128)
