## About this example

This examples shows the use of the _section_ directive in OpenMP.

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
Thread 0 of 4 calculates section 2
Thread 2 of 4 calculates section 1
Thread 1 of 4 calculates section 4
Thread 3 of 4 calculates section 3
```

## Exra Resources

* http://openmp.org/wp/resources/
* https://msdn.microsoft.com/en-us//library/8k4b1177.aspx


