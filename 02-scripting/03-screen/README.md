## About this example

This example shows how to use Screen is a full-screen software program that can be used to multiplexes a physical console between several processes. For this it is shown through a countdown program as you can exit and enter the same process.

## Requirements

You should have a compiler installed and Screen. Ubuntu Linux:

```bash
apt-get install build-essential cmake
```
apt-get install screen


## Run

Open a terminal and type:

```bash
sh run.sh

screen -ls ( list screen session)

## Output

A typical output should look like this one. 

There is a screen on:
        3119.Counter    (Detached)
```
## Run
screen -r Counter (attach session)

Press CTRL a + d (dettach session)


## Output

A typical output should look like this one. 
Second 120
Second 119
Second 118
Second 117 
Second 116 
Second 115 
Second 114 
Second 113 
Second 112 
Second 111 
Second 110 
Second 109 
Second 108 



[detached from 3119.Counter]


## Extra Resources
https://www.gnu.org/software/screen/manual/screen.html
```
