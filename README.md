# Clustering_algorithm
kmeans,kmeans-kernel,dbscan,kmeansPlusPlus, some clustering algorithm replementation

any clustering models can be add into the src/models/

run as script.sh as following:

``` sh
bash script.sh
```

each line of sh is showed as following:
``` sh
python main.py --data iris --type kmeans --nb_clusters 0
```
`nb_clustering` default is `0`, such that the number of clustering will be matched automatically, according to the sklearn dataset.


Each line of script can add one line of clustering result into the {data}_res.csv


# Requirements.txt
```
numpy==1.15.0
pandas==0.25.1
numpy==1.18.2
scikit_learn==0.22.2.post1

```

``` sh
pip install -r requirements.txt

```
