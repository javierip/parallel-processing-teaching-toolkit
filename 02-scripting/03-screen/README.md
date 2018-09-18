# Screen
If you see that your remote session in finished due inactivity, you can use this tool. Screen is a full-screen window manager that multiplexes a physical terminal between several processes.

## Basic Screen Usage
Below are the most basic steps for getting started with screen [courtesy of this website](https://linuxize.com/post/how-to-use-linux-screen/):

1. On the command prompt, type screen.
2. Run the desired program.
3. Use the key sequence Ctrl-a + Ctrl-d to detach from the screen session.
4. Reattach to the screen session by typing screen -r.

## About this example

This example shows how to use Screen is a full-screen software program that can be used to multiplexes a physical console between several processes. For this it is shown through a countdown program as you can exit and enter the same process.

## Requirements

You should have a compiler installed and Screen. Ubuntu Linux:

```bash
apt-get install build-essential cmake
apt-get install screen

```

## Run

Open a terminal and type:

```bash
sh run.sh

screen -ls ( list screen session)

```
## Output

A typical output should look like this one. 

```
There is a screen on:
        3119.Counter    (Detached)

```
## Run

```
screen -r Counter (attach session)

Press CTRL a + d (dettach session)

```
## Output

A typical output should look like this one. 

```
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

```

## Extra Resources
https://www.gnu.org/software/screen/manual/screen.html
```
