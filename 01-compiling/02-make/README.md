## About this example

This example shows a C program compilation and run using a Makefile.

## Requirements

You should have a compiler installed. Ubuntu Linux:

```bash
apt-get install build-essential
```

## Run

Open a terminal and type:

```bash
> sh run.sh
```

It will run `hello-pycuda.py` with 400 elements in each array.


## Output

A typical output should look like this one. 

```
compiling ..
gcc -o application main.c functions.c -I.
running..
Helo there!
```
