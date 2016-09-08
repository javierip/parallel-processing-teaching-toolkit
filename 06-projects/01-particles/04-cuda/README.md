## About this example

This is the CUDA implementation of the particles simulator.

## Requirements

You must have CUDA installed in your system. Look at the [CUDA section](../../../04-GPU-accelerators/01-cuda) for examples.

## Run

Open a terminal and type:

```bash
> sh run.sh
```

It will run the simulator using 100 particles and will generate an output file named _output-serial-100-particles.txt_.

##  Output
```
javier@orca:~ > sh run.sh 
-- The C compiler identification is GNU 4.8.5
-- The CXX compiler identification is GNU 4.9.3
-- Check for working C compiler: /usr/bin/cc
-- Check for working C compiler: /usr/bin/cc -- works
-- Detecting C compiler ABI info
-- Detecting C compiler ABI info - done
-- Check for working CXX compiler: /usr/bin/c++
-- Check for working CXX compiler: /usr/bin/c++ -- works
-- Detecting CXX compiler ABI info
-- Detecting CXX compiler ABI info - done
-- Found CUDA: /usr/local/cuda-7.5 (found version "7.5") 
-- Configuring done
-- Generating done
-- Build files have been written to: /home/javier/build
[ 50%] Building NVCC (Device) object CMakeFiles/./application-CUDA.dir//././application-CUDA_generated_common.cu.o
[100%] Building NVCC (Device) object CMakeFiles/./application-CUDA.dir//././application-CUDA_generated_main.cu.o
Scanning dependencies of target application-CUDA
Linking CXX executable ./application-CUDA
[100%] Built target ./application-CUDA
CPU-GPU copy time = 2.5e-05 seconds
n = 1000, simulation time = 0.407526 seconds
javier@orca:~ > 

```
