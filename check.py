import json
import nilearn
import numpy
import os

from nilearn import image
from tqdm import tqdm

missing_infos = {
                 'subject' : list(),
                 'sex' : list(),
                 'age' : list(),
                 'T1' : {'comprehension' : list(), 'production' : list(), 'other' : list(),'original_avg' : list()},
                 'T2' : {'comprehension' : list(), 'production' : list(), 'other' : list(), 'original_avg' : list()},
                 'T3' : {'comprehension' : list(), 'production' : list(), 'other' : list(), 'original_avg' : list()},
                 }

fortyone = list()
with open(os.path.join('dataset', 'data_41.tsv')) as i:
    for l_i, l in enumerate(i):
        line = l.strip().split('\t')
        if l_i == 0:
            header = line.copy()
            continue
        fortyone.append('{}.{}'.format(line[header.index('Clinic_ID')].strip(), line[header.index('Age')].strip()))
        for k in ['T1', 'T2', 'T3']:
            missing_infos[k]['original_avg'].append(line[header.index(k)])
assert len(fortyone) == 41

paths = list()
ps = json.load(open('patients_behav.csv'))
for sub_age in fortyone:
    age = int(sub_age.split('.')[1])
    sub = sub_age.split('.')[0]
    for p_i, p in enumerate(ps):
        patient = p['patientId'].strip()
        if sub != patient:
            continue
        if age != p['age']:
            continue
        if sub_age in missing_infos['subject']:
            print(p)
            continue
        #print(p['distanceMap'])
        print(p['lesion'])
        ### adding infos
        missing_infos['subject'].append(sub_age)
        missing_infos['sex'].append(p['sex'])
        missing_infos['age'].append(p['age'])
        for exp in p['experiments']:
            _ = exp['name2'][-1]
            missing_infos['T{}'.format(_)]['comprehension'].append(exp['LRSComp'])
            missing_infos['T{}'.format(_)]['production'].append(exp['LRSProd'])
            missing_infos['T{}'.format(_)]['other'].append(exp['LRS'])

        #path = os.path.join('lesions', p['lesion'])
        #assert os.path.exists(path)
        #paths.append(path)
import pdb; pdb.set_trace()

assert sorted(fortyone) == sorted(missing_infos['subject'])
with open('demo_behav_41.tsv', 'w') as o:
    for k in ['subject', 'sex', 'age']:
        o.write('{}\t'.format(k))
    for k in ['T1', 'T2', 'T3']:
        for k_two in ['comprehension', 'production', 'other', 'original_avg']:
            o.write('{}_{}\t'.format(k, k_two))
    o.write('\n')
    for s_i in range(len(missing_infos['subject'])):
        for k in ['subject', 'sex', 'age']:
            o.write('{}\t'.format(missing_infos[k][s_i]))
        for k in ['T1', 'T2', 'T3']:
            for k_two in ['comprehension', 'production', 'other', 'original_avg']:
                o.write('{}\t'.format(missing_infos[k][k_two][s_i]))
        o.write('\n')

