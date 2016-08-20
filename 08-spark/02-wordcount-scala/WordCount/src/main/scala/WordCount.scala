
import org.apache.spark.SparkContext
import org.apache.spark.SparkContext._
import org.apache.spark.SparkConf


object WordCount {
  def main(args: Array[String]): Unit = {
    val inputPath = args(0)
    val sc = new SparkContext()
    val lines = sc.textFile(inputPath)
    val wordCounts = lines.flatMap {line => line.split(" ")}
                          .map(word => (word, 1))
                          .reduceByKey(_ + _)
    wordCounts.collect().foreach(println)
    } 
}

