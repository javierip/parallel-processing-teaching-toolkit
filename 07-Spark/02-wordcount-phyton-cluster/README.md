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
1. sudo /usr/local/spark/sbin/start.all
2. sh run.sh


#Output

 INFO StandaloneSchedulerBackend: Connected to Spark cluster with app ID app-20160822095815-0002

 INFO Utils: Successfully started service 'org.apache.spark.network.netty.NettyBlockTransferService' on port 33983.

 INFO NettyBlockTransferService: Server created on localhost:33983

 INFO StandaloneAppClient$ClientEndpoint: Executor added: app-20160822095815-0002/0 on worker-20160822094931-localhost-46544 (localhost:46544) with 2 cores

 INFO BlockManagerMaster: Registering BlockManager BlockManagerId(driver, localhost, 33983)

 INFO StandaloneSchedulerBackend: Granted executor ID app-20160822095815-0002/0 on hostPort localhost:46544 with 2 cores, 1024.0 MB RAM

 INFO BlockManagerMasterEndpoint: Registering block manager localhost:33983 with 366.3 MB RAM, BlockManagerId(driver, localhost, 33983)

 INFO BlockManagerMaster: Registered BlockManager BlockManagerId(driver, localhost, 33983)

 INFO StandaloneAppClient$ClientEndpoint: Executor updated: app-20160822095815-0002/0 is now RUNNING

 INFO StandaloneSchedulerBackend: SchedulerBackend is ready for scheduling beginning after reached minRegisteredResourcesRatio: 0.0

 INFO SharedState: Warehouse path is 'file:/home/adrian/spark-warehouse'.

 INFO CoarseGrainedSchedulerBackend$DriverEndpoint: Registered executor NettyRpcEndpointRef(null) (localhost:35154) with ID 0

 INFO BlockManagerMasterEndpoint: Registering block manager localhost:39242 with 366.3 MB RAM, BlockManagerId(0, localhost, 39242)

 INFO FileSourceStrategy: Pruning directories with: 

 INFO FileSourceStrategy: Post-Scan Filters: 

 INFO FileSourceStrategy: Pruned Data Schema: struct<value: string>

 INFO FileSourceStrategy: Pushed Filters: 

 INFO MemoryStore: Block broadcast_0 stored as values in memory (estimated size 264.2 KB, free 366.0 MB)

 INFO MemoryStore: Block broadcast_0_piece0 stored as bytes in memory (estimated size 23.1 KB, free 366.0 MB)

 INFO BlockManagerInfo: Added broadcast_0_piece0 in memory on localhost:33983 (size: 23.1 KB, free: 366.3 MB)

 INFO SparkContext: Created broadcast 0 from javaToPython at NativeMethodAccessorImpl.java:-2

 INFO FileSourceStrategy: Planning scan with bin packing, max size: 4194304 bytes, open cost is considered as scanning 4194304 bytes.

 INFO CodeGenerator: Code generated in 248.650031 ms

 INFO SparkContext: Starting job: collect at /usr/local/spark/examples/src/main/python/wordcount.py:40

 INFO DAGScheduler: Registering RDD 5 (reduceByKey at /usr/local/spark/examples/src/main/python/wordcount.py:39)

 INFO DAGScheduler: Got job 0 (collect at /usr/local/spark/examples/src/main/python/wordcount.py:40) with 1 output partitions

 INFO DAGScheduler: Final stage: ResultStage 1 (collect at /usr/local/spark/examples/src/main/python/wordcount.py:40)

 INFO DAGScheduler: Parents of final stage: List(ShuffleMapStage 0)

 INFO DAGScheduler: Missing parents: List(ShuffleMapStage 0)

 INFO DAGScheduler: Submitting ShuffleMapStage 0 (PairwiseRDD[5] at reduceByKey at /usr/local/spark/examples/src/main/python/wordcount.py:39), which has no missing parents

 INFO MemoryStore: Block broadcast_1 stored as values in memory (estimated size 12.9 KB, free 366.0 MB)

 INFO MemoryStore: Block broadcast_1_piece0 stored as bytes in memory (estimated size 7.5 KB, free 366.0 MB)

 INFO BlockManagerInfo: Added broadcast_1_piece0 in memory on localhost:33983 (size: 7.5 KB, free: 366.3 MB)

 INFO SparkContext: Created broadcast 1 from broadcast at DAGScheduler.scala:1012

 INFO DAGScheduler: Submitting 1 missing tasks from ShuffleMapStage 0 (PairwiseRDD[5] at reduceByKey at /usr/local/spark/examples/src/main/python/wordcount.py:39)

 INFO TaskSchedulerImpl: Adding task set 0.0 with 1 tasks

 INFO TaskSetManager: Starting task 0.0 in stage 0.0 (TID 0, localhost, partition 0, PROCESS_LOCAL, 5958 bytes)

 INFO CoarseGrainedSchedulerBackend$DriverEndpoint: Launching task 0 on executor id: 0 hostname: localhost.

 INFO BlockManagerInfo: Added broadcast_1_piece0 in memory on localhost:39242 (size: 7.5 KB, free: 366.3 MB)

 INFO BlockManagerInfo: Added broadcast_0_piece0 in memory on localhost:39242 (size: 23.1 KB, free: 366.3 MB)

 INFO TaskSetManager: Finished task 0.0 in stage 0.0 (TID 0) in 4029 ms on localhost (1/1)

 INFO TaskSchedulerImpl: Removed TaskSet 0.0, whose tasks have all completed, from pool 

 INFO DAGScheduler: ShuffleMapStage 0 (reduceByKey at /usr/local/spark/examples/src/main/python/wordcount.py:39) finished in 4,056 s

 INFO DAGScheduler: looking for newly runnable stages

 INFO DAGScheduler: running: Set()

 INFO DAGScheduler: waiting: Set(ResultStage 1)

 INFO DAGScheduler: failed: Set()

 INFO DAGScheduler: Submitting ResultStage 1 (PythonRDD[8] at collect at /usr/local/spark/examples/src/main/python/wordcount.py:40), which has no missing parents

 INFO MemoryStore: Block broadcast_2 stored as values in memory (estimated size 6.1 KB, free 366.0 MB)

 INFO MemoryStore: Block broadcast_2_piece0 stored as bytes in memory (estimated size 3.7 KB, free 366.0 MB)

 INFO BlockManagerInfo: Added broadcast_2_piece0 in memory on localhost:33983 (size: 3.7 KB, free: 366.3 MB)

 INFO SparkContext: Created broadcast 2 from broadcast at DAGScheduler.scala:1012

 INFO DAGScheduler: Submitting 1 missing tasks from ResultStage 1 (PythonRDD[8] at collect at /usr/local/spark/examples/src/main/python/wordcount.py:40)

 INFO TaskSchedulerImpl: Adding task set 1.0 with 1 tasks

 INFO TaskSetManager: Starting task 0.0 in stage 1.0 (TID 1, localhost, partition 0, NODE_LOCAL, 5286 bytes)

 INFO CoarseGrainedSchedulerBackend$DriverEndpoint: Launching task 1 on executor id: 0 hostname: localhost.

 INFO BlockManagerInfo: Added broadcast_2_piece0 in memory on localhost:39242 (size: 3.7 KB, free: 366.3 MB)

 INFO MapOutputTrackerMasterEndpoint: Asked to send map output locations for shuffle 0 to localhost:35154

 INFO MapOutputTrackerMaster: Size of output statuses for shuffle 0 is 145 bytes

 INFO TaskSetManager: Finished task 0.0 in stage 1.0 (TID 1) in 191 ms on localhost (1/1)

 INFO DAGScheduler: ResultStage 1 (collect at /usr/local/spark/examples/src/main/python/wordcount.py:40) finished in 0,193 s

 INFO TaskSchedulerImpl: Removed TaskSet 1.0, whose tasks have all completed, from pool 

 INFO DAGScheduler: Job 0 finished: collect at /usr/local/spark/examples/src/main/python/wordcount.py:40, took 4,552225 s

: 1

world: 2

Hello: 3


