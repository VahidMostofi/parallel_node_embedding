#!/bin/bash
#SBATCH --job-name=spark-node2vec      # create a short name for your job
#SBATCH --nodes=1                # node count
#SBATCH --ntasks-per-node=8      # number of tasks per node
#SBATCH --cpus-per-task=1        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem=0                 # memory per node
#SBATCH --partition=parallel

module purge

module load spark/2.2.0 python/anaconda3-2018.12

#conda activate pyspark2
#export PYSPARK_PYTHON=/home/vahid.mirzaebrahimmo/.conda/envs/pyspark2/bin/python
#spark-submit --total-executor-cores 60 --executor-memory 5G pi.py 100
spark-submit parallel_node_embedding_spark.py --batches 16 --dir ../data/livemocha --combinations max --voting --weighted


