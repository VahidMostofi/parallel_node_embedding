#!/bin/bash
export PYSPARK_PYTHON=/home/vahid.mirzaebrahimmo/.conda/envs/pyspark2/bin/python
spark-submit --executor-memory 22G --master spark://cn0588:7077 parallel_node_embedding_spark.py --batches 8 --dir ../data/livemocha --combinations max --wc 10 --wl 80 --dim 8
spark-submit --executor-memory 22G --master spark://cn0588:7077 parallel_node_embedding_spark.py --batches 8 --dir ../data/livemocha --combinations max --wc 10 --wl 80 --dim 64
spark-submit --executor-memory 22G --master spark://cn0588:7077 parallel_node_embedding_spark.py --batches 8 --dir ../data/livemocha --combinations max --wc 10 --wl 80 --dim 128
spark-submit --executor-memory 22G --master spark://cn0588:7077 parallel_node_embedding_spark.py --batches 8 --dir ../data/livemocha --combinations max --wc 5 --wl 40 --dim 8
spark-submit --executor-memory 22G --master spark://cn0588:7077 parallel_node_embedding_spark.py --batches 8 --dir ../data/livemocha --combinations max --wc 5 --wl 40 --dim 64
spark-submit --executor-memory 22G --master spark://cn0588:7077 parallel_node_embedding_spark.py --batches 8 --dir ../data/livemocha --combinations max --wc 5 --wl 40 --dim 128
spark-submit --executor-memory 22G --master spark://cn0588:7077 parallel_node_embedding_spark.py --batches 8 --dir ../data/livemocha --combinations max --wc 2 --wl 15 --dim 8
spark-submit --executor-memory 22G --master spark://cn0588:7077 parallel_node_embedding_spark.py --batches 8 --dir ../data/livemocha --combinations max --wc 2 --wl 15 --dim 64
spark-submit --executor-memory 22G --master spark://cn0588:7077 parallel_node_embedding_spark.py --batches 8 --dir ../data/livemocha --combinations max --wc 2 --wl 15 --dim 128
spark-submit --executor-memory 22G --master spark://cn0588:7077 parallel_node_embedding_spark.py --batches 64 --dir ../data/livemocha --combinations max --wc 10 --wl 80 --dim 8
spark-submit --executor-memory 22G --master spark://cn0588:7077 parallel_node_embedding_spark.py --batches 64 --dir ../data/livemocha --combinations max --wc 10 --wl 80 --dim 64
spark-submit --executor-memory 22G --master spark://cn0588:7077 parallel_node_embedding_spark.py --batches 64 --dir ../data/livemocha --combinations max --wc 10 --wl 80 --dim 128
spark-submit --executor-memory 22G --master spark://cn0588:7077 parallel_node_embedding_spark.py --batches 64 --dir ../data/livemocha --combinations max --wc 5 --wl 40 --dim 8
spark-submit --executor-memory 22G --master spark://cn0588:7077 parallel_node_embedding_spark.py --batches 64 --dir ../data/livemocha --combinations max --wc 5 --wl 40 --dim 64
spark-submit --executor-memory 22G --master spark://cn0588:7077 parallel_node_embedding_spark.py --batches 64 --dir ../data/livemocha --combinations max --wc 5 --wl 40 --dim 128
spark-submit --executor-memory 22G --master spark://cn0588:7077 parallel_node_embedding_spark.py --batches 64 --dir ../data/livemocha --combinations max --wc 2 --wl 15 --dim 8
spark-submit --executor-memory 22G --master spark://cn0588:7077 parallel_node_embedding_spark.py --batches 64 --dir ../data/livemocha --combinations max --wc 2 --wl 15 --dim 64
spark-submit --executor-memory 22G --master spark://cn0588:7077 parallel_node_embedding_spark.py --batches 64 --dir ../data/livemocha --combinations max --wc 2 --wl 15 --dim 128