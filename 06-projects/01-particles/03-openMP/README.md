## About this example

This is the OpenMP implementation of the particles simulator.

## Requirements

You should have a compiler installed. Ubuntu Linux:

```bash
sudo apt-get install build-essential
```

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
-- Try OpenMP C flag = [-fopenmp]
-- Performing Test OpenMP_FLAG_DETECTED
-- Performing Test OpenMP_FLAG_DETECTED - Success
-- Try OpenMP CXX flag = [-fopenmp]
-- Performing Test OpenMP_FLAG_DETECTED
-- Performing Test OpenMP_FLAG_DETECTED - Success
-- Found OpenMP: -fopenmp  
OPENMP FOUND
-- Configuring done
-- Generating done
-- Build files have been written to: /home/javier/build
Scanning dependencies of target application
[ 50%] Building CXX object CMakeFiles/application.dir/openmp.cpp.o
[100%] Building CXX object CMakeFiles/application.dir/common.cpp.o
Linking CXX executable application
[100%] Built target application
n = 100,threads = 8, simulation time = 1.39221 seconds, absmin = 0.796793, absavg = -nan
javier@orca:~ > 

```
