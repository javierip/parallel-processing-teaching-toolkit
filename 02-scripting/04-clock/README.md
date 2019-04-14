## About this example

This example shows how measure the time that took a program to run.

## Requirements

You should have a compiler installed. Ubuntu Linux:

```bash
apt-get install build-essential cmake
```

## Run

Open a terminal and type:

```bash
sh run.sh
```


## Output

A typical output should look like this one. 

```
-- The C compiler identification is GNU 7.3.0
-- The CXX compiler identification is GNU 7.3.0
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
-- Build files have been written to: /home/javier/11-code-hpc/parallel-processing-teaching-toolkit/02-scripting/04-clock/build
Scanning dependencies of target application
[ 50%] Building C object CMakeFiles/application.dir/main.c.o
[100%] Linking C executable application
[100%] Built target application
Elapsed: 0.135165 seconds

```
