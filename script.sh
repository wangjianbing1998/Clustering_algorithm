# nb_clusters==0 means the value of it will be automaticly getted
python main.py --data iris --type kmeans --nb_clusters 0
python main.py --data iris --type kmeans_kernel --sigma 4. --nb_clusters 0
python main.py --data iris --type dbscan --min_points 2 --eps .5
python main.py --data iris --type kmeansPP --nb_clusters 3



python main.py --data boston --type kmeans --nb_clusters 0
python main.py --data boston --type kmeans_kernel --sigma 4. --nb_clusters 0
python main.py --data boston --type dbscan --min_points 2 --eps .5
python main.py --data boston --type kmeansPP --nb_clusters 3


python main.py --data diabetes --type kmeans --nb_clusters 0
python main.py --data diabetes --type kmeans_kernel --sigma 4. --nb_clusters 0
python main.py --data diabetes --type dbscan --min_points 2 --eps .5
python main.py --data diabetes --type kmeansPP --nb_clusters 3



python main.py --data breast_cancer --type kmeans --nb_clusters 0
python main.py --data breast_cancer --type kmeans_kernel --sigma 4. --nb_clusters 0
python main.py --data breast_cancer --type dbscan --min_points 2 --eps .5
python main.py --data breast_cancer --type kmeansPP --nb_clusters 3

