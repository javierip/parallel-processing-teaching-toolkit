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
    $ sh run.sh
```

#Output

The output shows many lines of status information of the run. Between these lines is the result of the count of words


	INFO Utils: Successfully started service 'SparkUI' on port 4040.
	INFO SparkUI: Bound SparkUI to 0.0.0.0, and started at http://localhost:4040
	INFO SparkContext: Added file file:/usr/local/spark/examples/src/main/python/wordcount.py at file:/usr/local/spark/examples/src/main/python/wordcount.py with timestamp 1471871541824
	INFO SparkContext: Starting job: collect at /usr/local/spark/examples/src/main/python/wordcount.py:40
	INFO DAGScheduler: Registering RDD 5 (reduceByKey at /usr/local/spark/examples/src/main/python/wordcount.py:39)
	INFO DAGScheduler: Got job 0 (collect at /usr/local/spark/examples/src/main/python/wordcount.py:40) with 1 output partitions
	INFO DAGScheduler: Final stage: ResultStage 1 (collect at /usr/local/spark/examples/src/main/python/wordcount.py:40)
	INFO DAGScheduler: Submitting 1 missing tasks from ShuffleMapStage 0 (PairwiseRDD[5] at reduceByKey at /usr/local/spark/examples/src/main/python/wordcount.py:39)
	INFO TaskSchedulerImpl: Adding task set 0.0 with 1 tasks
	INFO TaskSetManager: Starting task 0.0 in stage 0.0 (TID 0, localhost, partition 0, PROCESS_LOCAL, 5969 bytes)
	INFO Executor: Fetching file:/usr/local/spark/examples/src/main/python/wordcount.py with timestamp 1471871541824
	INFO TaskSetManager: Finished task 0.0 in stage 0.0 (TID 0) in 1007 ms on localhost (1/1)
	INFO DAGScheduler: ShuffleMapStage 0 (reduceByKey at /usr/local/spark/examples/src/main/python/wordcount.py:39) finished in 1,056 s
	INFO DAGScheduler: Submitting 1 missing tasks from ResultStage 1 (PythonRDD[8] at collect at /usr/local/spark/examples/src/main/python/wordcount.py:40)
	INFO Executor: Finished task 0.0 in stage 1.0 (TID 1). 1909 bytes result sent to driver
	INFO DAGScheduler: ResultStage 1 (collect at /usr/local/spark/examples/src/main/python/wordcount.py:40) finished in 0,122 s
	INFO DAGScheduler: Job 0 finished: collect at /usr/local/spark/examples/src/main/python/wordcount.py:40, took 1,425430 s

	
	world: 2
	Hello: 3


# References

 * [Apache Spark 2.0.0 - Quick Start](http://spark.apache.org/docs/latest/quick-start.html)
 * [Apache Spark 2.0.0 - Spark Standalone](http://spark.apache.org/docs/latest/spark-standalone.html)
 * [Apache Spark 2.0.0 - Submitting Applications](http://spark.apache.org/docs/latest/submitting-applications.html)

