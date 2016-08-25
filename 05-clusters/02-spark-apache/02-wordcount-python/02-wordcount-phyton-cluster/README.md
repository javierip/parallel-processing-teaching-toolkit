#About

Counts the number of words that are in the file word.txt

##Requirements
* Download and install Java from https://www.java.com/es/download/ or from distribution Ubuntu:
```
	$ sudo apt install icedtea-8-plugin openjdk-8-jre
```
* Download Spark from http://spark.apache.org/downloads.html and install:
```
	$ sudo tar xvfz spark.version.tgz -C   /usr/local/
	$ sudo ln -s /usr/local/spark.version   /usr/local/spark
```
#Run

Open a terminal and type:
```
	$ sudo /usr/local/spark/sbin/start-all.sh
	$ sh run.sh
```

#Output

The output shows many lines of status information of the run. Between these lines is the result of the count of words



	INFO StandaloneSchedulerBackend: Connected to Spark cluster with app ID app-20160822095815-0002
	INFO NettyBlockTransferService: Server created on localhost:33983
	INFO StandaloneAppClient$ClientEndpoint: Executor added: app-20160822095815-0002/0 on worker-20160822094931-localhost-46544 (localhost:46544) with 2 cores
	INFO StandaloneAppClient$ClientEndpoint: Executor updated: app-20160822095815-0002/0 is now RUNNING
	INFO StandaloneSchedulerBackend: SchedulerBackend is ready for scheduling beginning after reached minRegisteredResourcesRatio: 0.0
	INFO SparkContext: Starting job: collect at /usr/local/spark/examples/src/main/python/wordcount.py:40
	INFO DAGScheduler: Registering RDD 5 (reduceByKey at /usr/local/spark/examples/src/main/python/wordcount.py:39)
	INFO DAGScheduler: Got job 0 (collect at /usr/local/spark/examples/src/main/python/wordcount.py:40) with 1 output partitions
	INFO DAGScheduler: Final stage: ResultStage 1 (collect at /usr/local/spark/examples/src/main/python/wordcount.py:40)
	INFO DAGScheduler: Submitting ShuffleMapStage 0 (PairwiseRDD[5] at reduceByKey at /usr/local/spark/examples/src/main/python/wordcount.py:39), which has no missing parents
	INFO DAGScheduler: Submitting 1 missing tasks from ShuffleMapStage 0 (PairwiseRDD[5] at reduceByKey at /usr/local/spark/examples/src/main/python/wordcount.py:39)

	INFO TaskSetManager: Starting task 0.0 in stage 0.0 (TID 0, localhost, partition 0, PROCESS_LOCAL, 5958 bytes)
	INFO CoarseGrainedSchedulerBackend$DriverEndpoint: Launching task 0 on executor id: 0 hostname: localhost.
	INFO TaskSetManager: Finished task 0.0 in stage 0.0 (TID 0) in 4029 ms on localhost (1/1)
	INFO DAGScheduler: ShuffleMapStage 0 (reduceByKey at /usr/local/spark/examples/src/main/python/wordcount.py:39) finished in 4,056 s
	INFO DAGScheduler: Submitting ResultStage 1 (PythonRDD[8] at collect at /usr/local/spark/examples/src/main/python/wordcount.py:40), which has no missing parents
	INFO DAGScheduler: Submitting 1 missing tasks from ResultStage 1 (PythonRDD[8] at collect at /usr/local/spark/examples/src/main/python/wordcount.py:40)

	INFO TaskSetManager: Starting task 0.0 in stage 1.0 (TID 1, localhost, partition 0, NODE_LOCAL, 5286 bytes)
	INFO CoarseGrainedSchedulerBackend$DriverEndpoint: Launching task 1 on executor id: 0 hostname: localhost.
	INFO MapOutputTrackerMasterEndpoint: Asked to send map output locations for shuffle 0 to localhost:35154
	INFO TaskSetManager: Finished task 0.0 in stage 1.0 (TID 1) in 191 ms on localhost (1/1)
	INFO DAGScheduler: ResultStage 1 (collect at /usr/local/spark/examples/src/main/python/wordcount.py:40) finished in 0,193 s
	INFO DAGScheduler: Job 0 finished: collect at /usr/local/spark/examples/src/main/python/wordcount.py:40, took 4,552225 s


*world: 2*
*Hello: 3*

# References

 * [Apache Spark 2.0.0 - Quick Start](http://spark.apache.org/docs/latest/quick-start.html)
 * [Apache Spark 2.0.0 - Spark Cluster](http://spark.apache.org/docs/latest/cluster-overview.html)
 * [Apache Spark 2.0.0 - Submitting Applications](http://spark.apache.org/docs/latest/submitting-applications.html)

