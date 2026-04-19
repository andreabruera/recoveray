import random
import matplotlib
import numpy
import os

from matplotlib import pyplot

data = list()
with open(os.path.join('dataset','data_41.tsv')) as i:
    for l_i, l in enumerate(i):
        line = [w.replace(',', '.') for w in l.strip().split('\t')]
        if l_i == 0:
            header = line.copy()[2:]
            continue
        data.append(numpy.array(line[2:], dtype=numpy.float32))
data = numpy.array(data)

out = 'distributions'
os.makedirs(out, exist_ok=True)

relevant_vars = set(['_'.join(w.split('_')[:-1]) for w in header if '_T' in w and len(w)>5])
'''
print(relevant_vars)

for var_i, var in enumerate(header):
    fig, ax = pyplot.subplots(constrained_layout=True)
    ax.hist(data[:, var_i], bins=10)
    pyplot.savefig(os.path.join(out, '{}.jpg'.format(var)))
    pyplot.clf()
    pyplot.close()
'''

cmap = matplotlib.colormaps['hsv'].resampled(41)

for var in relevant_vars:
    if '_to_' in var:
        folder = 'connectivity'
    else:
        folder = 'activity'
    out = os.path.join('distributions', folder)
    os.makedirs(out, exist_ok=True)
    fig, ax = pyplot.subplots(constrained_layout=True)
    corrs = [random.randrange(0, 20)*0.01 for _ in range(data.shape[0])]
    for t in [1, 2, 3]:
        curr_var = '{}_T{}'.format(var, t)
        var_i = header.index(curr_var)
        ax.violinplot(
                     dataset=data[:, var_i],
                     positions=[t],
                     side='low',
                     showmeans=True,
                     )
        ax.scatter(
                   [t+corr for corr in corrs],
                   data[:, var_i],
                   color='grey'
                   )
        if t > 1:
            old_var = '{}_T{}'.format(var, t-1)
            old_var_i = header.index(old_var)
            for corr_i, corr in enumerate(corrs):
                ax.plot(
                        [t-1+corr, t+corr],
                        [data[corr_i, old_var_i], data[corr_i, var_i]],
                        #color='grey',
                        color=cmap(corr_i),
                        )

    pyplot.savefig(os.path.join(out, '{}.jpg'.format(var)))
    pyplot.clf()
    pyplot.close()


