#About

Counts the number of words that are in the file word.txt

##Requirements
* Download and install Java from https://www.java.com/es/download/ or from distribution Ubuntu:

1. apt install icedtea-8-plugin openjdk-8-jre

* Download Spark from http://spark.apache.org/downloads.html and install:

1. sudo tar xvfz spark.version.tgz -C   /usr/local/
2. sudo ln -s /usr/local/spark.version   /usr/local/spark

#Run

Open a terminal and type:

sh run.sh


#Output
 INFO Utils: Successfully started service 'sparkDriver' on port 37563.
 INFO SparkEnv: Registering MapOutputTracker
 INFO SparkEnv: Registering BlockManagerMaster
 INFO DiskBlockManager: Created local directory at /tmp/blockmgr-778abbeb-aec3-4068-be42-3070c3ee1818
 INFO MemoryStore: MemoryStore started with capacity 366.3 MB
 INFO SparkEnv: Registering OutputCommitCoordinator
 INFO Utils: Successfully started service 'SparkUI' on port 4040.
 INFO SparkUI: Bound SparkUI to 0.0.0.0, and started at http://localhost:4040
 INFO Utils: Copying /usr/local/spark/examples/src/main/python/wordcount.py to /tmp/spark-044fb414-44f7-4332-b00e-7cbcaf8c4bae/userFiles-f207fc77-30d8-4daa-9f44-14a239f1b513/wordcount.py
 INFO SparkContext: Added file file:/usr/local/spark/examples/src/main/python/wordcount.py at file:/usr/local/spark/examples/src/main/python/wordcount.py with timestamp 1471871541824
 INFO Executor: Starting executor ID driver on host localhost
 INFO Utils: Successfully started service 'org.apache.spark.network.netty.NettyBlockTransferService' on port 34283.
 INFO NettyBlockTransferService: Server created on localhost:34283
 INFO BlockManagerMaster: Registering BlockManager BlockManagerId(driver, localhost, 34283)
 INFO BlockManagerMasterEndpoint: Registering block manager localhost:34283 with 366.3 MB RAM, BlockManagerId(driver, localhost, 34283)
 INFO BlockManagerMaster: Registered BlockManager BlockManagerId(driver, localhost, 34283)
 INFO SharedState: Warehouse path is 'file:/home/adrian/spark-warehouse'.
 INFO FileSourceStrategy: Pruning directories with: 
 INFO FileSourceStrategy: Post-Scan Filters: 
 INFO FileSourceStrategy: Pruned Data Schema: struct<value: string>
 INFO FileSourceStrategy: Pushed Filters: 
 INFO MemoryStore: Block broadcast_0 stored as values in memory (estimated size 264.2 KB, free 366.0 MB)
 INFO MemoryStore: Block broadcast_0_piece0 stored as bytes in memory (estimated size 23.1 KB, free 366.0 MB)
 INFO BlockManagerInfo: Added broadcast_0_piece0 in memory on localhost:34283 (size: 23.1 KB, free: 366.3 MB)
 INFO SparkContext: Created broadcast 0 from javaToPython at NativeMethodAccessorImpl.java:-2
 INFO FileSourceStrategy: Planning scan with bin packing, max size: 4194335 bytes, open cost is considered as scanning 4194304 bytes.
 INFO CodeGenerator: Code generated in 246.771243 ms
 INFO SparkContext: Starting job: collect at /usr/local/spark/examples/src/main/python/wordcount.py:40
 INFO DAGScheduler: Registering RDD 5 (reduceByKey at /usr/local/spark/examples/src/main/python/wordcount.py:39)
 INFO DAGScheduler: Got job 0 (collect at /usr/local/spark/examples/src/main/python/wordcount.py:40) with 1 output partitions
 INFO DAGScheduler: Final stage: ResultStage 1 (collect at /usr/local/spark/examples/src/main/python/wordcount.py:40)
 INFO DAGScheduler: Parents of final stage: List(ShuffleMapStage 0)
 INFO DAGScheduler: Missing parents: List(ShuffleMapStage 0)
 INFO DAGScheduler: Submitting ShuffleMapStage 0 (PairwiseRDD[5] at reduceByKey at /usr/local/spark/examples/src/main/python/wordcount.py:39), which has no missing parents
 INFO MemoryStore: Block broadcast_1 stored as values in memory (estimated size 12.9 KB, free 366.0 MB)
 INFO MemoryStore: Block broadcast_1_piece0 stored as bytes in memory (estimated size 7.5 KB, free 366.0 MB)
 INFO BlockManagerInfo: Added broadcast_1_piece0 in memory on localhost:34283 (size: 7.5 KB, free: 366.3 MB)
 INFO SparkContext: Created broadcast 1 from broadcast at DAGScheduler.scala:1012
 INFO DAGScheduler: Submitting 1 missing tasks from ShuffleMapStage 0 (PairwiseRDD[5] at reduceByKey at /usr/local/spark/examples/src/main/python/wordcount.py:39)
 INFO TaskSchedulerImpl: Adding task set 0.0 with 1 tasks
 INFO TaskSetManager: Starting task 0.0 in stage 0.0 (TID 0, localhost, partition 0, PROCESS_LOCAL, 5969 bytes)
 INFO Executor: Running task 0.0 in stage 0.0 (TID 0)
 INFO Executor: Fetching file:/usr/local/spark/examples/src/main/python/wordcount.py with timestamp 1471871541824
 INFO Utils: /usr/local/spark/examples/src/main/python/wordcount.py has been previously copied to /tmp/spark-044fb414-44f7-4332-b00e-7cbcaf8c4bae/userFiles-f207fc77-30d8-4daa-9f44-14a239f1b513/wordcount.py
 INFO FileScanRDD: Reading File path: file:///home/adrian/word.txt, range: 0-31, partition values: [empty row]
 INFO CodeGenerator: Code generated in 16.189185 ms
 INFO PythonRunner: Times: total = 361, boot = 258, init = 99, finish = 4
 INFO Executor: Finished task 0.0 in stage 0.0 (TID 0). 2161 bytes result sent to driver
 INFO TaskSetManager: Finished task 0.0 in stage 0.0 (TID 0) in 1007 ms on localhost (1/1)
 INFO TaskSchedulerImpl: Removed TaskSet 0.0, whose tasks have all completed, from pool 
 INFO DAGScheduler: ShuffleMapStage 0 (reduceByKey at /usr/local/spark/examples/src/main/python/wordcount.py:39) finished in 1,056 s
 INFO DAGScheduler: looking for newly runnable stages
 INFO DAGScheduler: running: Set()
 INFO DAGScheduler: waiting: Set(ResultStage 1)
 INFO DAGScheduler: failed: Set()
 INFO DAGScheduler: Submitting ResultStage 1 (PythonRDD[8] at collect at /usr/local/spark/examples/src/main/python/wordcount.py:40), which has no missing parents
 INFO MemoryStore: Block broadcast_2 stored as values in memory (estimated size 6.1 KB, free 366.0 MB)
 INFO MemoryStore: Block broadcast_2_piece0 stored as bytes in memory (estimated size 3.7 KB, free 366.0 MB)
 INFO BlockManagerInfo: Added broadcast_2_piece0 in memory on localhost:34283 (size: 3.7 KB, free: 366.3 MB)
 INFO SparkContext: Created broadcast 2 from broadcast at DAGScheduler.scala:1012
 INFO DAGScheduler: Submitting 1 missing tasks from ResultStage 1 (PythonRDD[8] at collect at /usr/local/spark/examples/src/main/python/wordcount.py:40)
 INFO TaskSchedulerImpl: Adding task set 1.0 with 1 tasks
 INFO TaskSetManager: Starting task 0.0 in stage 1.0 (TID 1, localhost, partition 0, ANY, 5297 bytes)
 INFO Executor: Running task 0.0 in stage 1.0 (TID 1)
 INFO ShuffleBlockFetcherIterator: Getting 1 non-empty blocks out of 1 blocks
 INFO ShuffleBlockFetcherIterator: Started 0 remote fetches in 9 ms
 INFO PythonRunner: Times: total = 43, boot = -553, init = 596, finish = 0
 INFO Executor: Finished task 0.0 in stage 1.0 (TID 1). 1909 bytes result sent to driver
 INFO DAGScheduler: ResultStage 1 (collect at /usr/local/spark/examples/src/main/python/wordcount.py:40) finished in 0,122 s
 INFO TaskSetManager: Finished task 0.0 in stage 1.0 (TID 1) in 126 ms on localhost (1/1)
 INFO TaskSchedulerImpl: Removed TaskSet 1.0, whose tasks have all completed, from pool 
 INFO DAGScheduler: Job 0 finished: collect at /usr/local/spark/examples/src/main/python/wordcount.py:40, took 1,425430 s
: 1
world: 2
Hello: 3

