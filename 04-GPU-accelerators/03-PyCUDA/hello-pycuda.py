import pycuda.autoinit
import pycuda.driver as drv
import numpy
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("arrayLength",
                    help="the length of the array",
                    type=int)
args = parser.parse_args()

from pycuda.compiler import SourceModule
mod = SourceModule("""
__global__ void multiply_them(float *dest, float *a, float *b)
{
  const int i = threadIdx.x;
  dest[i] = a[i] * b[i];

}
""")

multiply_them = mod.get_function("multiply_them")

a = numpy.random.randn(args.arrayLength).astype(numpy.float32)
b = numpy.random.randn(args.arrayLength).astype(numpy.float32)

output = numpy.zeros_like(a)
multiply_them(
    drv.Out(output), drv.In(a), drv.In(b),
    block=(args.arrayLength, 1, 1), grid=(1, 1)
)

print 'A = {}'.format(a)
print 'B = {}'.format(b)
print 'Output = A * B = {}'.format(output)
