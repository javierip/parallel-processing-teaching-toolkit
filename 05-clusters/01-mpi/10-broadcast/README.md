##  Compile
Open a terminal and type:
```bash
> sh run.sh 
```

##  Output
```
javier@delfin:~/ > sh run.sh 
-- The C compiler identification is GNU 4.8.4
-- The CXX compiler identification is GNU 4.8.4
-- Check for working C compiler: /usr/bin/cc
-- Check for working C compiler: /usr/bin/cc -- works
-- Detecting C compiler ABI info
-- Detecting C compiler ABI info - done
-- Check for working CXX compiler: /usr/bin/c++
-- Check for working CXX compiler: /usr/bin/c++ -- works
-- Detecting CXX compiler ABI info
-- Detecting CXX compiler ABI info - done
-- Found MPI_C: /usr/lib/libmpi.so;/usr/lib/x86_64-linux-gnu/libdl.so;/usr/lib/x86_64-linux-gnu/libhwloc.so  
-- Found MPI_CXX: /usr/lib/libmpi_cxx.so;/usr/lib/libmpi.so;/usr/lib/x86_64-linux-gnu/libdl.so;/usr/lib/x86_64-linux-gnu/libhwloc.so  
-- Configuring done
-- Generating done
-- Build files have been written to: /home/javier//build
Scanning dependencies of target application-MPI
[100%] Building C object CMakeFiles/application-MPI.dir/main.c.o
Linking C executable application-MPI
[100%] Built target application-MPI
3
Process 0 got 3
Process 2 got 3
Process 1 got 3
Process 4 got 3
Process 3 got 3
Process 7 got 3
Process 5 got 3
Process 6 got 3
-1
Process 2 got -1
Process 0 got -1
Process 6 got -1
Process 4 got -1
Process 5 got -1
Process 3 got -1
Process 1 got -1
Process 7 got -1
javier@delfin:~/ > 

```


