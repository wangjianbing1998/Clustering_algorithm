evaluation_type = ['normalized_mutual_info_score', 'v_measure_score', 'completeness_score', 'homogeneity_score',
                   'adjusted_mutual_info_score',
                   'mutual_info_score', 'adjusted_rand_score']


def squaredDistance(vec1, vec2):
    sum = 0
    dim = len(vec1)

    for i in range(dim):
        sum += (vec1[i] - vec2[i]) * (vec1[i] - vec2[i])

    return sum


def print_options(args, parser):
    message = ''
    message += '----------------- Options ---------------\n'
    for k, v in sorted(vars(args).items()):
        comment = ''
        default = parser.get_default(k)
        if v != default:
            comment = '\t[default: %s]' % str(default)
        message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
    message += '----------------- End -------------------'
    print(message)


def evaluation(targets, pred):
    d={}
    for e in evaluation_type:
        exec('from sklearn import metrics')
        d[e]=eval(f'metrics.{e}(targets, pred)')

    return d
