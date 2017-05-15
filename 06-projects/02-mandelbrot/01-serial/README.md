## About this example

This is the serial implementation of the Mandelbrot set.

## Requirements

You should have a compiler installed. Ubuntu Linux:

```bash
sudo apt-get install cmake
```

## Run

Open a terminal and type:

```bash
sh run.sh
```

It will compute and save the Mandelbrot set. The output will be located at ./build/output.ppm

##  Output
```
javier@orca:~ > sh run.sh 
-- The C compiler identification is GNU 5.4.0
-- The CXX compiler identification is GNU 5.4.0
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
-- Configuring done
-- Generating done
-- Build files have been written to: /home/javier/01-code/parallel-code-examples/06-projects/02-mandelbrot/01-serial/build
Scanning dependencies of target application
[ 50%] Building C object CMakeFiles/application.dir/main.c.o
[100%] Linking C executable application
[100%] Built target application

javier@orca:~ >ls build
application  CMakeCache.txt  CMakeFiles  cmake_install.cmake  Makefile  output.ppm

```
