## About this example

This is the serial implementation of the particles simulator, similar to the [previous one](../01-serial-plain), but it adds a correctness check.

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
-- Configuring done
-- Generating done
-- Build files have been written to: /home/javier/build
Scanning dependencies of target application
[ 50%] Building CXX object CMakeFiles/application.dir/serial.cpp.o
[100%] Building CXX object CMakeFiles/application.dir/common.cpp.o
Linking CXX executable application
[100%] Built target application
n = 100, simulation time = 0.114029 seconds, absmin = 0.806317, absavg = 0.955301
javier@orca:~ > 

```
