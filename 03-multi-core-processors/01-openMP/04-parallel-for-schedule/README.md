## About this example

This example performs a vector addition using a [loop schedule](https://software.intel.com/en-us/articles/openmp-loop-scheduling).

## Requirements

You should have a compiler installed. Ubuntu Linux:

```bash
sudo apt-get install qt-sdk
```

## Run

Open a terminal and type:

```bash
> sh run.sh
```

## Output

A typical output should look like this one:

```
Static scheduling
Thread 0 of 4 is calculating the iteration 0
Thread 0 of 4 is calculating the iteration 1
Thread 0 of 4 is calculating the iteration 8
Thread 0 of 4 is calculating the iteration 9
Thread 1 of 4 is calculating the iteration 2
Thread 1 of 4 is calculating the iteration 3
Thread 3 of 4 is calculating the iteration 6
Thread 3 of 4 is calculating the iteration 7
Thread 2 of 4 is calculating the iteration 4
Thread 2 of 4 is calculating the iteration 5
Dynamic scheduling
Thread 0 of 4 is calculating the iteration 0
Thread 0 of 4 is calculating the iteration 1
Thread 2 of 4 is calculating the iteration 2
Thread 2 of 4 is calculating the iteration 3
Thread 0 of 4 is calculating the iteration 8
Thread 0 of 4 is calculating the iteration 9
Thread 3 of 4 is calculating the iteration 4
Thread 3 of 4 is calculating the iteration 5
Thread 1 of 4 is calculating the iteration 6
Thread 1 of 4 is calculating the iteration 7

```

## Exra Resources

The official site of OpenMP has a [list of resources](http://openmp.org/wp/resources/) you can look at to learn more.
