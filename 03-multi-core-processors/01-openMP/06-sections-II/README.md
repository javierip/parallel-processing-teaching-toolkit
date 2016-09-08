## About this example

This examples shows the use of the _section_ directive in OpenMP.

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
Thread 0 of 4 calculates i = 0 (section 1)
Thread 0 of 4 calculates i = 1 (section 1)
Thread 0 of 4 calculates i = 2 (section 1)
Thread 0 of 4 calculates i = 3 (section 1)
Thread 0 of 4 calculates i = 4 (section 1)
Thread 0 of 4 calculates i = 5 (section 1)
Thread 0 of 4 calculates i = 6 (section 1)
Thread 0 of 4 calculates i = 7 (section 1)
Thread 0 of 4 calculates i = 8 (section 1)
Thread 0 of 4 calculates i = 9 (section 1)
Thread 3 of 4 calculates i = 0 (section 2)
Thread 3 of 4 calculates i = 1 (section 2)
Thread 3 of 4 calculates i = 2 (section 2)
Thread 3 of 4 calculates i = 3 (section 2)
Thread 3 of 4 calculates i = 4 (section 2)
Thread 3 of 4 calculates i = 5 (section 2)
Thread 3 of 4 calculates i = 6 (section 2)
Thread 3 of 4 calculates i = 7 (section 2)
Thread 3 of 4 calculates i = 8 (section 2)
Thread 3 of 4 calculates i = 9 (section 2)
```

## Exra Resources

* http://openmp.org/wp/resources/
* https://msdn.microsoft.com/en-us//library/8k4b1177.aspx
