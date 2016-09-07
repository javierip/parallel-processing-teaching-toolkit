## About this example

This example is similar to the [first "Hello World" example](../01-hello-openMP), but it runs the same code using different amount of threads.

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
Setting a fixed number of threads. In this case 8
The total number of threads is 1
Hello! my ID is 1
Hello! my ID is 2
Hello! my ID is 3
Hello! my ID is 4
Hello! my ID is 5
Hello! my ID is 6
Hello! my ID is 7
Hello! my ID is 0
I am the thread 0 and the total numer is 8
Now we use 5 threads
Total number of threads 1
Hello! my ID is 1
Hello! my ID is 4
Hello! my ID is 3
Hello! my ID is 2
Hello! my ID is 0
I am the thread 0 and the total numer is 5
```

## Exra Resources

The official site of OpenMP has a [list of resources](http://openmp.org/wp/resources/) you can look at to learn more.

