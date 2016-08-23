#!/bin/bash
/usr/local/spark/bin/spark-submit --master  local /usr/local/spark/examples/src/main/python/wordcount.py word.txt
