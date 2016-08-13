#About

Counts the number of words that are in the file word.txt

#Run

Open a terminal and type:

sh run.sh


#Output

INFO TaskSchedulerImpl: Adding task set 1.0 with 1 tasks
INFO TaskSetManager: Starting task 0.0 in stage 1.0 (TID 1, localhost, partition 0, ANY, 5297 bytes)
INFO Executor: Running task 0.0 in stage 1.0 (TID 1)
INFO ShuffleBlockFetcherIterator: Getting 1 non-empty blocks out of 1 blocks
INFO ShuffleBlockFetcherIterator: Started 0 remote fetches in 9 ms
INFO PythonRunner: Times: total = 45, boot = -513, init = 557, finish = 1
INFO Executor: Finished task 0.0 in stage 1.0 (TID 1). 1909 bytes result sent to driver
INFO TaskSetManager: Finished task 0.0 in stage 1.0 (TID 1) in 110 ms on localhost (1/1)
INFO TaskSchedulerImpl: Removed TaskSet 1.0, whose tasks have all completed, from pool 
INFO DAGScheduler: ResultStage 1 (collect at /usr/local/spark/examples/src/main/python/wordcount.py:40) finished in 0,111 s
INFO DAGScheduler: Job 0 finished: collect at /usr/local/spark/examples/src/main/python/wordcount.py:40, took 1,429788 s

: 1
world: 2
Hello: 3

