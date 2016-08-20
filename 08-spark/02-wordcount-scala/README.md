# About

Counts the number of words that are in the file word.txt

# Requeriments

You can run the program written in scala locally (standalone) or on a cluster where we need to have previously run one spark master and a worker to take care of running the program counter words. 
To launch the master run:

  $ /usr/local/spark/sbin/start-master.sh

To verify that the master is running we can go to page http://localhost:8080/. This page allows you to monitor the status of the master and keeps track of the workers, of the applications that are running and completed

To launch a worker run:
 
  $ /usr/local/spark/bin/spark-class org.apache.spark.deploy.worker.Worker spark://localhost:7077

Then we can verify on the page that it is running and is assigned an worker identifier (Id), an IP address and port, and resources are available to run jobs as the number of cores and memory.

# Compille

To run the code WordCount.scala is necessary to generate a .jar file. For this we use the tool sbt (Simple Build Tool](). sbt can be downloaded from [www.scala-sbt.org/download.html]. To install run:

  echo "deb https://dl.bintray.com/sbt/debian /" | sudo tee -a /etc/apt/sources.list.d/sbt.list
  sudo apt-key adv --keyserver hkp://keyserver.ubuntu.com:80 --recv 642AC823
  sudo apt-get update
  sudo apt-get install sbt

The structure of directories and files required by sbt to compile the .scala file and get the .jar file is shown running the command tree:

  $ tree WordCount
  WordCount
  ├── src
  │   └── main
  │       └── scala
  │           └── WordCount.scala
  ├── target
  │   └── scala-2.11
  │       └── word-count_2.11-1.0.jar
  └── wordcount.sbt  

Finally we run sbt to build the package and we get a sucess at the end:

  $ cd WorkCount
  $ sbt package
  [info] Set current project to Word Count (in build file:/home/user/WordCount/)
  [info] Compiling 1 Scala source to /home/user/WordCount/target/scala-2.11/classes...
  [info] Packaging /home/user/WordCount/target/scala-2.11/word-count_2.11-1.0.jar ...
  [info] Done packaging.
  [success] Total time: 27 s, completed Aug 17, 2016 7:06:58 PM

# Run

Spark provides the tool spark-submit to submitting jobs to a cluster. This tool can also be used to run jobs locally. Here are some examples of runs:

 * Locally (standalone)
  $ /usr/local/spark/bin/spark-submit --class WordCount --master local target/scala-2.11/word-count_2.11-1.0.jar ../word.txt

 * Locally with 4 Cores 
  $ /usr/local/spark/bin/spark-submit --class WordCount --master local[4] target/scala-2.11/simple-project_2.11-1.0.jar ../word.txt

 * In a cluster using a worker (master and worker running on the local machine) 
  $ /usr/local/spark/bin/spark-submit --class WordCount --master spark://localhost:7077 target/scala-2.11/word-count_2.11-1.0.jar ../word.txt

# Output

The output shows many lines of status information of the run. Between these lines is the result of the count of words
  ....
  (Hello,3)
  (World,2)
  ....

# References

 * [Apache Spark 2.0.0 - Quick Start](http://spark.apache.org/docs/latest/quick-start.html)
 * [Apache Spark 2.0.0 - Spark Programming Guide](http://spark.apache.org/docs/latest/programming-guide.html)
 * [Apache Spark 2.0.0 - Submitting Applications](http://spark.apache.org/docs/latest/submitting-applications.html)
 * Big Data Analytics with Spark. Mohammed Guller. Apress, 2015.
 * Learning Spark. Holden Karau, Andy Konwinski, Patrick Wendell, and Matei Zaharia. O'Reilly. 2015.
