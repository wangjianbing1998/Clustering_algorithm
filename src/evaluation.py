from math import log, sqrt


def count_occurrence(list):
    d = {}
    for i in list:
        if i in d:
            d[i] += 1
        else:
            d[i] = 1
    return d


def cal_entropy(assignment):
    occ = count_occurrence(assignment)  # get # of data points in each cluster, dictionary
    n = float(sum(occ.values()))  # number of data points
    h = 0  # entropy of cluster
    for id in occ:
        p = occ[id] / n
        if p != 0:
            h += p * log(p)
    return -h


def purity(groundtruthAssignment, algorithmAssignment):
    purity = 0
    ids = sorted(set(algorithmAssignment))
    matching = 0
    for id in ids:
        indices = [i for i, j in enumerate(algorithmAssignment) if j == id]
        cluster = [groundtruthAssignment[i] for i in indices]
        occ = count_occurrence(cluster)
        matching += max(occ.values())
    purity = matching / float(len(groundtruthAssignment))
    print('Purity: ', end='')
    return purity


def NMI(groundtruthAssignment, algorithmAssignment):
    h_c = cal_entropy(algorithmAssignment)
    h_t = cal_entropy(groundtruthAssignment)
    occ_c = count_occurrence(algorithmAssignment)
    occ_t = count_occurrence(groundtruthAssignment)  # get occurrence: for the probability of cluster T_id
    n_t = float(sum(occ_t.values()))
    ids_c = sorted(set(algorithmAssignment))
    ids_t = sorted(set(groundtruthAssignment))

    cp = [(i, j) for i in ids_c for j in ids_t]
    p = dict(list(zip(cp, [0] * len(cp))))

    for (i, j) in zip(algorithmAssignment, groundtruthAssignment):
        p[(i, j)] += 1

    mi = 0
    for c in ids_c:
        for t in ids_t:
            if p[(c, t)] != 0:
                mi += (p[(c, t)] / n_c) * log((p[(c, t)] / n_c) / ((occ_c[c] / n_c) * (occ_t[t] / n_t)))
    NMI = mi / sqrt(float(h_c * h_t))

    print('NMI: ', end='')

    return NMI
