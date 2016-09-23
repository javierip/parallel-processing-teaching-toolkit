#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import, print_function
import numpy as np
import pyopencl as cl

a_host = np.random.rand(50000).astype(np.float32)
b_host = np.random.rand(50000).astype(np.float32)

ctx = cl.create_some_context()
queue = cl.CommandQueue(ctx)

mf = cl.mem_flags
a_device = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=a_host)
b_device = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=b_host)

prg = cl.Program(ctx, """
__kernel void sum(__global const float *a_device, __global const float *b_device, __global float *res_device) {
  int gid = get_global_id(0);
  res_device[gid] = a_device[gid] + b_device[gid];
}
""").build()

res_device = cl.Buffer(ctx, mf.WRITE_ONLY, a_host.nbytes)
prg.sum(queue, a_host.shape, None, a_device, b_device, res_device)

res_host = np.empty_like(a_host)
cl.enqueue_copy(queue, res_host, res_device)

# Check on CPU with Numpy:
print(res_host - (a_host + b_host))
print(np.linalg.norm(res_host - (a_host + b_host)))
