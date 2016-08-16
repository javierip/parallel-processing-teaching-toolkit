## About this example

This example uses the Python's [multiprocessing library](https://docs.python.org/2/library/multiprocessing.html) to perform a long array summation.
It includes a serial example using just a `for` loop and the parallel example using a pool of 4 workers.
## Requirements

* Pyhton > 2.6

## Run

Open a terminal and type:

```bash
> sh run.sh
```

## Output

A tipical output should look like this one:

```
Serial Example:
Result: 500000000500000000
Serial Execution Time: 30.7276070118s
Multiprocessing Example:
Result: 500000000500000000
Multiprocessing Execution Time: 16.964548111s
```

## Extra Resources

For futher information, check out the following links:
* https://docs.python.org/2/library/multiprocessing.html
* http://kmdouglass.github.io/posts/learning-pythons-multiprocessing-module.html
* http://spartanideas.msu.edu/2014/06/20/an-introduction-to-parallel-programming-using-pythons-multiprocessing-module/
