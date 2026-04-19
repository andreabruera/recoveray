import numpy
import os

n_subjects = 41
### start
with open(os.path.join('dataset', 'data_41.tsv')) as i:
    for l_i, l in enumerate(i):
        line = l.split('\t')
        line[-1] = line[-1].strip()
        if l_i == 0:
            #header = [w for w in line]
            header = list()
            for d in line:
                if len(d) == 5 and d[0]=='T' and d[-2]=='T':
                    d = d.replace('_', '-')
                header.append(d)
            #raw_data = {h : numpy.zeros(shape=(n_subjects,)) for h in header[2:] if h!='L_SMA' and 'adjusted' not in h}
            raw_data = {h : numpy.zeros(shape=(n_subjects,)) for h in header[2:] if 'adjusted' not in h}
            continue
        for d in raw_data.keys():
            val = line[header.index(d)].replace(',', '.')
            if val == '':
                #raw_data[d].append(numpy.nan)
                raw_data[d][l_i-1] = numpy.nan
            else:
                #raw_data[d].append(float(val))
                raw_data[d][l_i-1] = float(val)
assert l_i == n_subjects

import pdb; pdb.set_trace()
