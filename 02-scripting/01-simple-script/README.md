## About this example

This example shows how to handle arguments in a C program.

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
-- Configuring done
-- Generating done
-- Build files have been written to: /home/javier/03-code/parallel-code-examples/02-scripting/01-simple-script/build
Scanning dependencies of target application
[100%] Building C object CMakeFiles/application.dir/main.c.o
Linking C executable application
[100%] Built target application
Hello there !, there are 1 arguments
Argument 0 is ./application
Hello there !, there are 5 arguments
Argument 0 is ./application
Argument 1 is 2
Argument 2 is 5
Argument 3 is text
Argument 4 is another_text
```
