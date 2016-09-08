## About this example

This is the MPI implementation of the particles simulator.

## Requirements

You must have MPI installed. Look at the [MPI section](../../../05-clusters/01-mpi) for example.

## Run

Open a terminal and type:

```bash
> sh run.sh
```

It will run the simulator using 8 nodes, 100 particles and will generate an output file named _output-serial-100-particles.txt_.

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
-- Found MPI_C: /usr/lib/openmpi/lib/libmpi.so;/usr/lib/x86_64-linux-gnu/libdl.so;/usr/lib/x86_64-linux-gnu/libhwloc.so  
-- Found MPI_CXX: /usr/lib/openmpi/lib/libmpi_cxx.so;/usr/lib/openmpi/lib/libmpi.so;/usr/lib/x86_64-linux-gnu/libdl.so;/usr/lib/x86_64-linux-gnu/libhwloc.so  
-- Configuring done
-- Generating done
-- Build files have been written to: /home/javier//build
Scanning dependencies of target application
[ 50%] Building CXX object CMakeFiles/application.dir/mpi.cpp.o
[100%] Building CXX object CMakeFiles/application.dir/common.cpp.o
Linking CXX executable application
[100%] Built target application
n = 100, simulation time = 0.26358 seconds, absmin = 0.778965, absavg = 0.955897
javier@perca:~/ > 

```
