## About this example

This examples shows the use of the _critical_ directive in OpenMP.

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
Using critical
Thread 0 is accessing value 1
Thread 1 is accessing value 2
Thread 2 is accessing value 3
Thread 3 is accessing value 4
Final value of the addition is 4
Not using critical
Thread 1 is accessing value 1
Thread 3 is accessing value 2
Thread 2 is accessing value 2
Thread 0 is accessing value 3
Final value of the addition is 3

```

## Exra Resources

* http://openmp.org/wp/resources/
* https://msdn.microsoft.com/en-us/library/b38674ky.aspx

