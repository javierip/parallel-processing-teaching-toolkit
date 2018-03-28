# -*- coding: utf-8 -*-

from __future__ import print_function

import numpy as np

from pycuda import driver, compiler, gpuarray
import time

# -- initialize the device
import pycuda.autoinit

kernel_source_code = """

__global__ void markThreadID(float *a, float *id_blocks_x, float *id_blocks_y, float *id_threads_x, float *id_threads_y, float *id_cell)
{
    int tx = threadIdx.x + blockIdx.x * blockDim.x;
    int ty = threadIdx.y + blockIdx.y * blockDim.y;

    id_blocks_x[ty * %(MATRIX_SIZE)s + tx] = blockIdx.x;
    id_blocks_y[ty * %(MATRIX_SIZE)s + tx] = blockIdx.y;
    
    id_threads_x[ty * %(MATRIX_SIZE)s + tx] = threadIdx.x;
    id_threads_y[ty * %(MATRIX_SIZE)s + tx] = threadIdx.y;
    
    id_cell[ty * %(MATRIX_SIZE)s + tx] = ty * %(MATRIX_SIZE)s + tx;

}

"""


def show_values(matrix_size, threads_per_block):
    a_cpu = np.random.randn(matrix_size, matrix_size).astype(np.float32)

    # transfer host (CPU) memory to device (GPU) memory
    a_gpu = gpuarray.to_gpu(a_cpu)
    id_groups_x = gpuarray.empty((matrix_size, matrix_size), np.float32)
    id_groups_y = gpuarray.empty((matrix_size, matrix_size), np.float32)
    id_threads_x = gpuarray.empty((matrix_size, matrix_size), np.float32)
    id_threads_y = gpuarray.empty((matrix_size, matrix_size), np.float32)
    id_cell = gpuarray.empty((matrix_size, matrix_size), np.float32)

    blocks = (threads_per_block, 1, 1)

    blocks_per_side = int(matrix_size / threads_per_block)

    if (blocks_per_side * threads_per_block) < matrix_size:
        blocks_per_side = blocks_per_side + 1

    grid = (blocks_per_side, matrix_size, 1)

    print("Blocks: ", blocks)
    print("Grid: ", grid)

    kernel_code = kernel_source_code % {'MATRIX_SIZE': matrix_size, 'BLOCK_SIZE': threads_per_block}

    compiled_kernel = compiler.SourceModule(kernel_code)

    kernel_binary = compiled_kernel.get_function("markThreadID")

    kernel_binary(
        # inputs
        a_gpu,
        # outputs
        id_groups_x, id_groups_y, id_threads_x, id_threads_y, id_cell,
        block=blocks,
        grid=grid
    )

    id_blocks_x_cpu = id_groups_x.get()
    id_blocks_y_cpu = id_groups_y.get()
    id_threads_x_cpu = id_threads_x.get()
    id_threads_y_cpu = id_threads_y.get()
    id_cell_cpu = id_cell.get()

    print("id_blocks_x_cpu")
    print(id_blocks_x_cpu)

    print("id_blocks_y_cpu")
    print(id_blocks_y_cpu)

    print("id_threads_x_cpu")
    print(id_threads_x_cpu)

    print("id_threads_y_cpu")
    print(id_threads_y_cpu)

    print("id_cell_cpu")
    print(id_cell_cpu)


if __name__ == "__main__":
    matrix_size = 8

    show_values(matrix_size, 2)
    show_values(matrix_size, 4)

    matrix_size = 9

    show_values(matrix_size, 2)
    show_values(matrix_size, 4)
