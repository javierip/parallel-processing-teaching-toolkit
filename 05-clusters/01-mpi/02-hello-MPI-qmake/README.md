## About this example

This example shows how to run a MPI program on a single computer using QMake.

## Requirements
 
 * OpenMPI
 * QT SDK


## Run

Open a terminal and type:

```bash
sh run.sh
```


## Output

A typical output should look like this one. 
```
javier@perca:~/ > sh run.sh 
mpicc -c -I/usr/lib/openmpi/include -I/usr/lib/openmpi/include/openmpi -pthread -O2 -Wall -W -D_REENTRANT -DQT_NO_DEBUG -DQT_GUI_LIB -DQT_CORE_LIB -DQT_SHARED -I/usr/share/qt4/mkspecs/linux-g++-64 -I. -I/usr/include/qt4/QtCore -I/usr/include/qt4/QtGui -I/usr/include/qt4 -I. -I. -o main.o main.c
mpicxx -pthread -L/usr//lib -L/usr/lib/openmpi/lib -lmpi_cxx -lmpi -ldl -lhwloc -Wl,-O1 -o application-MPI main.o    -L/usr/lib/x86_64-linux-gnu -lQtGui -lQtCore -lpthread 
Hello world from process 1 of 8
Hello world from process 2 of 8
Hello world from process 5 of 8
Hello world from process 0 of 8
Hello world from process 3 of 8
Hello world from process 4 of 8
Hello world from process 7 of 8
Hello world from process 6 of 8
javier@perca:~/ > 

```


