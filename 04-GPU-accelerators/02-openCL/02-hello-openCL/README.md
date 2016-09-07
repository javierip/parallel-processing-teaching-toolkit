## About this example

This example shows how to run and compile a simple OpenCL program using CMake.

## Requirements

CUDA Toolkit and proper Drivers.

## Run

Open a terminal and type:

```bash
sh run.sh
```


## Output

A typical output should look like this one. 

```
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
-- Found OpenCL: /usr/lib/x86_64-linux-gnu/libOpenCL.so  
-- Configuring done
-- Generating done
-- Build files have been written to: /home/javier/03-code/parallel-code-examples/04-GPU-accelerators/02-openCL/02-hello-openCL/build
Scanning dependencies of target application-openCL
[100%] Building CXX object CMakeFiles/application-openCL.dir/main.cpp.o
Linking CXX executable application-openCL
[100%] Built target application-openCL
./application-openCL: /usr/local/cuda-7.5/lib64/libOpenCL.so.1: no version information available (required by ./application-openCL)
0 3 6 9 12 15 18 21 24 27 30 33 36 39 42 45 48 51 54 57 60 63 66 ...
Executed program succesfully.
```

## Extra Resources

 * [OpenCL Programming Guide 1.2 Examples](https://github.com/bgaster/opencl-book-samples).
 * [NVIDIA toolkit documentation](https://developer.nvidia.com/cuda-toolkit).



