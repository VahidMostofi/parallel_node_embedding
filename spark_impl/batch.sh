#!/bin/bash
export PYSPARK_PYTHON=/home/vahid.mirzaebrahimmo/.conda/envs/pyspark2/bin/python
# spark-submit --executor-memory 22G --master spark://cn0588:7077 parallel_node_embedding_spark.py --batches 32 --dir ../data/livemocha --combinations 2 --wc 5 --wl 40 --dim 64
# spark-submit --executor-memory 22G --master spark://cn0588:7077 parallel_node_embedding_spark.py --batches 32 --dir ../data/livemocha --combinations 8 --wc 5 --wl 40 --dim 64

# spark-submit --executor-memory 22G --master spark://cn0588:7077 parallel_node_embedding_spark.py --batches 32 --dir ../data/livemocha --combinations 16 --wc 5 --wl 40 --dim 64
# spark-submit --executor-memory 22G --master spark://cn0588:7077 parallel_node_embedding_spark.py --batches 32 --dir ../data/livemocha --combinations 24 --wc 5 --wl 40 --dim 64