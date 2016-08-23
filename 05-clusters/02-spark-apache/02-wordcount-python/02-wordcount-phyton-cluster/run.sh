#!/bin/bash
/usr/local/spark/bin/spark-submit --master spark://localhost:7077  /usr/local/spark/examples/src/main/python/wordcount.py word.txt
