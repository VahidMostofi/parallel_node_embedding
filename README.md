# Parallel Node Embedding
I'm gonna write about this project and how it works in near future(hopefuly).

### Short Version
Implementation of Node embedding on large graphs using Apache Spark. This implemnation uses Node2Vec but the approach could be used on any Node Embedding technique that works on graphs.
The results of this new and dsitributed embedding is evaluated by performing link prediction and comparing the results with the case of using only one single machine to do the similar task. 
(All these sounds very confusing, I'll definetly write a blog post about it)
Long story short, It does node embedding using Spark to makes it possible for even large graphs.
# parallel_node_embedding
