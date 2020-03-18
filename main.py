# coding=gbk
import os
import time

import pandas as pd

pd.set_option('expand_frame_repr', False)
pd.set_option('display.max_rows', 6)
pd.set_option('precision', 2)
import argparse

from sklearn.preprocessing import StandardScaler

from src import models
from src.utils import print_options, evaluation, evaluation_type

parser = argparse.ArgumentParser(description='Clustering Algorithm')
parser.add_argument('--data', default='iris', type=str, required=True,
                    choices=['iris', 'boston', 'diabetes', 'breast_cancer'],
                    help='the dataset for clustering')
parser.add_argument('--type', default='kmeans', type=str, required=True,
                    choices=['kmeans', 'kmeans_kernel', 'dbscan', 'kmeansPP'],
                    help='the type of clustering algorithm')

parser.add_argument('--nb_clusters', default=0, type=int, required=False,
                    help='optional number of clustering for kmeans algorithm, default 0 means auto get')
parser.add_argument('--sigma', default=4., type=float, required=False,
                    help='optional sigma for kmeans-kernel algorithm')
parser.add_argument('--min_points', default=2, type=int, required=False,
                    help='optional min points for dbscan algorithm')
parser.add_argument('--eps', default=.5, type=float, required=False,
                    help='optional eps for dbscan algorithm')

parser.add_argument('--outf', default='res.csv', type=str, required=False,
                    help='optional result file path')
args = parser.parse_args()

print_options(args, parser)

data, groundtruth = None, None
exec('from sklearn import datasets')
exec(f'data, groundtruth = datasets.load_{args.data}(return_X_y=True)')

if args.nb_clusters == 0:
    args.nb_clusters = len(set(groundtruth))
print(f'nb_clusters={args.nb_clusters}')

if args.type == 'kmeans' and not args.nb_clusters \
        or args.type == 'kmeans_kernel' and not args.sigma and not args.nb_clusters \
        or args.type == 'kmeansPP' and not args.sigma and not args.nb_clusters \
        or args.type == 'dbscan' and not args.min_points and not args.eps:
    raise ValueError(f'optional param must be contained when algorithm is {args.type}')

data = StandardScaler().fit_transform(data)

end = time.time()
results = models.__dict__[args.type](args, data)
t = time.time()

if not os.path.exists(args.data + '_' + args.outf):
    res = pd.DataFrame(
        columns=['type', 'time', *evaluation_type])
else:
    res = pd.read_csv(args.data + '_' + args.outf)
dd = pd.DataFrame(evaluation(groundtruth, results), index=[0])
dd['type'] = args.type
dd['time'] = t
res = res.append(dd, ignore_index=True)
print(res)
res = res[['type', 'time', *evaluation_type]]
res.to_csv(args.data + '_' + args.outf, index=False)

# print(purity(groundtruth, np.array(results)))
# print(NMI(groundtruth, results))
