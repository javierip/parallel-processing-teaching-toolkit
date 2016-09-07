## About this example

This example shows how to run a MPI Scatter directive.

## Requirements
 
 * OpenMPI


## Run

Open a terminal and type:

```bash
sh run.sh
```


## Output

A typical output should look like this one. 
```
javier@delfin:~/ > sh run.sh 
-- The C compiler identification is GNU 4.8.4
-- The CXX compiler identification is GNU 4.8.4
-- Check for working C compiler: /usr/bin/cc
-- Check for working C compiler: /usr/bin/cc -- works
-- Detecting C compiler ABI info
-- Detecting C compiler ABI info - done
-- Check for working CXX compiler: /usr/bin/c++
-- Check for working CXX compiler: /usr/bin/c++ -- works
-- Detecting CXX compiler ABI info
-- Detecting CXX compiler ABI info - done
-- Found MPI_C: /usr/lib/libmpi.so;/usr/lib/x86_64-linux-gnu/libdl.so;/usr/lib/x86_64-linux-gnu/libhwloc.so  
-- Found MPI_CXX: /usr/lib/libmpi_cxx.so;/usr/lib/libmpi.so;/usr/lib/x86_64-linux-gnu/libdl.so;/usr/lib/x86_64-linux-gnu/libhwloc.so  
-- Configuring done
-- Generating done
-- Build files have been written to: /home/javier//build
Scanning dependencies of target application-MPI
[100%] Building C object CMakeFiles/application-MPI.dir/main.c.o
Linking C executable application-MPI
[100%] Built target application-MPI
rank= 0  Results: 1.000000 2.000000 3.000000 4.000000
rank= 2  Results: 9.000000 10.000000 11.000000 12.000000
rank= 1  Results: 5.000000 6.000000 7.000000 8.000000
rank= 3  Results: 13.000000 14.000000 15.000000 16.000000
Must specify 4 processors. Terminating.
Must specify 4 processors. Terminating.
Must specify 4 processors. Terminating.
Must specify 4 processors. Terminating.
Must specify 4 processors. Terminating.
javier@delfin:~/ > 

```


