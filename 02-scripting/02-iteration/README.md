## About this example

This example shows how to pass arguments to a C program and how to change the values of the shell variables.

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
[100%] Building C object CMakeFiles/application.dir/main.c.o
Linking C executable application
[100%] Built target application
Hello there !, there are 2 arguments
Argument 0 is ./application
Argument 1 is 1
Hello there !, there are 2 arguments
Argument 0 is ./application
Argument 1 is 2
Hello there !, there are 2 arguments
Argument 0 is ./application
Argument 1 is 4
Hello there !, there are 2 arguments
Argument 0 is ./application
Argument 1 is 8
Hello there !, there are 5 arguments
Argument 0 is ./application
Argument 1 is 2
Argument 2 is 5
Argument 3 is text
Argument 4 is another_text

```
