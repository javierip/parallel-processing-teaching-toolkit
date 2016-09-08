## About this example

This example performs a vector multiplication using OpenMP.

## Requirements

You should have a compiler installed. Ubuntu Linux:

```bash
sudo apt-get install cmake
```

## Run

Open a terminal and type:

```bash
> sh run.sh
```

## Output

A typical output should look like this one:

```
Thread 0 of 4 calculates i = 0
Thread 0 of 4 calculates i = 1
Thread 0 of 4 calculates i = 2
Thread 2 of 4 calculates i = 6
Thread 2 of 4 calculates i = 7
Thread 3 of 4 calculates i = 8
Thread 3 of 4 calculates i = 9
Thread 1 of 4 calculates i = 3
Thread 1 of 4 calculates i = 4
Thread 1 of 4 calculates i = 5
The content of vector A is:
 285  330  375  420  465  510  555  600  645  690
```

## Exra Resources

The official site of OpenMP has a [list of resources](http://openmp.org/wp/resources/) you can look at to learn more.
