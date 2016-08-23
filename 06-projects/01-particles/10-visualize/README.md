##  Compile
Open a terminal and type:
```bash
> sh run.sh 
```


##  Output
```
javier@perca:~/ > sh run.sh 
-- The C compiler identification is GNU 4.8.4
-- The CXX compiler identification is GNU 4.8.4
-- Check for working C compiler: /usr/bin/cc
-- Check for working C compiler: /usr/bin/cc -- works
-- Detecting C compiler ABI info
-- Detecting C compiler ABI info - done
-- Detecting C compile features
-- Detecting C compile features - done
-- Check for working CXX compiler: /usr/bin/c++
-- Check for working CXX compiler: /usr/bin/c++ -- works
-- Detecting CXX compiler ABI info
-- Detecting CXX compiler ABI info - done
-- Detecting CXX compile features
-- Detecting CXX compile features - done
-- Found CUDA: /usr/local/cuda-7.5 (found version "7.5") 
CMake Warning (dev) at /usr/share/cmake-3.2/Modules/FindCUDA.cmake:1561 (add_executable):
  Policy CMP0037 is not set: Target names should not be reserved and should
  match a validity pattern.  Run "cmake --help-policy CMP0037" for policy
  details.  Use the cmake_policy command to set the policy and suppress this
  warning.

  The target name "./application-CUDA" is reserved or not valid for certain
  CMake features, such as generator expressions, and may result in undefined
  behavior.
Call Stack (most recent call first):
  CMakeLists.txt:8 (CUDA_ADD_EXECUTABLE)
This warning is for project developers.  Use -Wno-dev to suppress it.

-- Configuring done
-- Generating done
-- Build files have been written to: /home/javier//build
[ 50%] Building NVCC (Device) object CMakeFiles/application-CUDA.dir/__/common/application-CUDA_generated_common.cu.o
[100%] Building NVCC (Device) object CMakeFiles/application-CUDA.dir/application-CUDA_generated_main.cu.o
Scanning dependencies of target application-CUDA
Linking CXX executable ./application-CUDA
[100%] Built target ./application-CUDA
CPU-GPU copy time = 4.5e-05 seconds
n = 1000, simulation time = 0.397745 seconds
javier@perca:~/ > 

```