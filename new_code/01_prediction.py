import numpy
import os
import pickle
import random
import re
import scipy
import sklearn

from sklearn import linear_model
from tqdm import tqdm

def plot_corr_mtrx(data, dimensions):
    ### looking at general correlations to check if things make sense
    corrs = scipy.stats.spearmanr(data).statistic
    assert corrs.shape == (len(dimensions), len(dimensions))

    fig, ax = pyplot.subplots(
                              figsize=(20, 20),
                                )
    ax.imshow(
              corrs,
              cmap='coolwarm',
              vmin=-1,
              vmax=1.,
              )
    pyplot.hlines(y=[1.5, 4.5, 7.5, 11.5, 17.5, 23.5, 29.5, 38.5, 47.5],
                  xmin=-.5,
                  xmax=len(dimensions)-.5,
                  linewidth=3,
                  color='white',
                  )
    pyplot.vlines(x=[1.5, 4.5, 7.5, 11.5, 17.5, 23.5, 29.5, 38.5, 47.5],
                  ymin=-.5,
                  ymax=len(dimensions)-.5,
                  linewidth=3,
                  color='white',
                  )
    for x in range(corrs.shape[0]):
        for y in range(corrs.shape[1]):
            if x == y:
                continue
            ax.text(
                    x=x,
                    y=y,
                    s=round(corrs[x, y], 2),
                    fontsize=10,
                    ha='center',
                    va='center',
                    color='white',
                    )
    pyplot.yticks(ticks=range(len(dimensions)), labels=dimensions)
    pyplot.xticks(ticks=range(len(dimensions)), labels=dimensions, rotation=45, ha='right')

    ax.spines[['left', 'right', 'bottom', 'top']].set_visible(False)
    pyplot.savefig(
                   os.path.join('plots', 'check_correlations_activity.jpg'),
                   pad_inches=0,
    )

### reading data
base_folder = 'dataset'

data = numpy.zeros(shape=(47, 57))
dimensions = ['age', 'sex', 'acute_dps', 'subacute_dps', 'early-chronic_dps', 'acute_score', 'subacute_score', 'early-chronic_score']

behavioural_f = os.path.join(base_folder, 'demo_behavioural', 'selected_subjects_closest-phase-mixed_corrected.tsv')
sub_check = list()
with open(behavioural_f) as i:
    for l_i, l in enumerate(i):
        line = l.strip().split('\t')
        if l_i == 0:
            header = line.copy()
            continue
        sub = int(re.sub('\D', '', line[header.index('bids_subject')]))
        sub_check.append(sub)
        assert sub >= 1 and sub <= 47
        age = int(line[header.index('age')])
        data[sub-1, 0] = age
        sex = -1 if line[header.index('sex')]=='M' else 1
        data[sub-1, 1] = sex
        acute_dps = float(line[header.index('acute_dps')])
        data[sub-1, 2] = acute_dps
        subacute_dps = float(line[header.index('subacute_dps')])
        data[sub-1, 3] = subacute_dps
        early_chronic_dps = float(line[header.index('early-chronic_dps')])
        data[sub-1, 4] = early_chronic_dps
        acute_score = float(line[header.index('acute_performance')])
        data[sub-1, 5] = acute_score
        subacute_score = float(line[header.index('subacute_performance')])
        data[sub-1, 6] = subacute_score
        early_chronic_score = float(line[header.index('early-chronic_performance')])
        data[sub-1, 7] = early_chronic_score
for _ in range(1, 48):
    assert _ in sub_check

lesion_f = os.path.join(base_folder, 'lesions', 'aphasia_recovery_47-lesion-affection.tsv')
sub_check = list()
with open(lesion_f) as i:
    for l_i, l in enumerate(i):
        line = l.strip().split('\t')
        if l_i == 0:
            header = line.copy()
            rel_keys = [h for h in header if 'lesion' in h]
            assert len(rel_keys) == 4
            dimensions.extend(rel_keys)
            continue

        sub = int(re.sub('\D', '', line[header.index('bids_subject')]))
        sub_check.append(sub)
        assert sub >= 1 and sub <= 47
        rel_keys = [h for h in header if 'lesion' in h]
        assert len(rel_keys) == 4
        for k in rel_keys:
            f_idx = header.index(k)
            mtrx_idx = dimensions.index(k)
            data[sub-1, mtrx_idx] = float(line[f_idx])
for _ in range(1, 48):
    assert _ in sub_check

#activity_f = 'aphasia_recovery_47-trimmed20mean-activity-allregressors.tsv'
activity_f = 'aphasia_recovery_47-mean-activity-allregressors.tsv'
#activity_f = 'aphasia_recovery_47-top10-activity-allregressors.tsv'
sub_check = list()
with open(os.path.join(base_folder, 'functional', activity_f)) as i:
    for l_i, l in enumerate(i):
        line = l.strip().split('\t')
        if l_i == 0:
            header = line.copy()
            for timepoint in ['acute', 'subacute', 'early-chronic']:
                rel_keys = [h for h in header if '_{}_'.format(timepoint) in h]
                assert len(rel_keys) == 6
                dimensions.extend(rel_keys)
            ### connectivity is not there
            assert len(dimensions) == data.shape[-1]-27
            continue

        assert len(dimensions) == data.shape[-1]-27
        sub = int(re.sub('\D', '', line[header.index('bids_subject')]))
        sub_check.append(sub)
        assert sub >= 1 and sub <= 47
        for timepoint in ['acute', 'subacute', 'early-chronic']:
            rel_keys = [h for h in header if '_{}_'.format(timepoint) in h]
            assert len(rel_keys) == 6
            for k in rel_keys:
                f_idx = header.index(k)
                mtrx_idx = dimensions.index(k)
                data[sub-1, mtrx_idx] = float(line[f_idx])
for _ in range(1, 48):
    assert _ in sub_check

connectivity_f = 'DCM_connectivity-b_formodel.tsv'
sub_check = list()
with open(os.path.join(base_folder, 'functional', connectivity_f)) as i:
    for l_i, l in enumerate(i):
        line = l.strip().split('\t')
        assert len(line) == 28
        if l_i == 0:
            header = line.copy()
            for timepoint in ['acute', 'subacute', 'early-chronic']:
                rel_keys = [h for h in header if '_{}_'.format(timepoint) in h]
                assert len(rel_keys) == 9
                dimensions.extend(rel_keys)
            assert len(dimensions) == data.shape[-1]
            continue

        assert len(dimensions) == data.shape[-1]
        sub = int(re.sub('\D', '', line[header.index('bids_subject')]))
        sub_check.append(sub)
        assert sub >= 1 and sub <= 47
        for timepoint in ['acute', 'subacute', 'early-chronic']:
            rel_keys = [h for h in header if '_{}_'.format(timepoint) in h]
            assert len(rel_keys) == 9
            for k in rel_keys:
                f_idx = header.index(k)
                mtrx_idx = dimensions.index(k)
                data[sub-1, mtrx_idx] = float(line[f_idx])
for _ in range(1, 48):
    assert _ in sub_check

#plot_corr_mtrx(data, dimensions)

n_subjects = 47
iterations = 1001
folds = 50
proportion = 0.2
test_size = int(47*proportion)

### using the exact same splits in all cases
seed = 138
random.seed(seed)
test_sets = list()
train_sets = list()
sets = {
        'train' : list(),
        'test' : list(),
        }
for __ in range(folds):
    iter_items = list(range(n_subjects))
    iter_test_size = int(n_subjects*proportion)
    test_items = random.sample(iter_items, k=iter_test_size)
    iter_test_items = test_items + random.choices(test_items, k=test_size-iter_test_size)
    train_items = [k for k in iter_items if k not in test_items]
    iter_train_items = train_items + random.choices(train_items, k=47-test_size-len(train_items))
    for sub in iter_test_items:
        assert sub not in iter_train_items
    sets['train'].append(iter_train_items)
    sets['test'].append(iter_test_items)
for k, v in sets.items():
    assert len(v) == folds

case_targets = {
           'ability':  ['acute', 'subacute', 'early-chronic'],
           'improvement':  ['acute2subacute', 'acute2early-chronic', 'subacute2early-chronic'],
          }

alphas=(
        1e-10, 1e-9, 1e-8, 1e-7, 1e-6,
        1e-5, 1e-4, 1e-3, 1e-2, 1e-1,
        1, 1e1, 1e2, 1e3, 1e4, 1e5,
        1e6, 1e7, 1e8, 1e9, 1e10
        )

details = {
          'traditional' : {
                           'modalities' : ['age',
                                          #'sex',
                                          'lesion',
                                          'acute_score',
                                          'subacute_score',
                                          'early-chronic_score',
                                          ],
                           #'confounds' :  [h for h in dimensions if 'activity' in h] + [h for h in dimensions if 'lesion' in h] + ['age', 'sex'],
                           'confounds' : [h for h in dimensions if 'connectivity' in h] + [h for h in dimensions if 'activity' in h] + [h for h in dimensions if 'lesion' in h] + ['age']
                            # + ['{}_score'.format(t) for t in case_targets['ability']]
                            ,
                           'povs' : ['all'],
                          },
          'functional' : {
                           'modalities' : [
                                          'activity',
                                          'connectivity',
                                          ],
                           #'confounds' :  ['age', 'sex'] + [h for h in dimensions if 'lesion' in h],
                           'confounds' :  ['age'] + [h for h in dimensions if 'lesion' in h],
                           'povs' : ['acute', 'subacute', 'early-chronic'],
                          },
            }

### full
all_results = dict()

for analysis_type in [
                      'functional',
                      'traditional',
                      ]:
    all_results[analysis_type] = {'details' : details[analysis_type]}
    modalities = details[analysis_type]['modalities']
    confounds = details[analysis_type]['confounds']
    povs = details[analysis_type]['povs']
    ### full model
    ### results for targets (always 3)

    results = {m : {k : numpy.zeros(shape=(3, len(povs), iterations, folds)) for k in modalities} for m in ['ability', 'improvement']}
    weights = {m : dict() for m in ['ability', 'improvement']}
    weights_names = {m : dict() for m in ['ability', 'improvement']}
    raw_ps = list()
    t_maxes = list()

    for case in results.keys():
        targets = case_targets[case]
        for targ_idx, targ in enumerate(targets):
            for pov_idx, pov in enumerate(povs):
                ### not interested in predicting the past from the future...
                if analysis_type == 'functional':
                    if case == 'ability':
                        if pov_idx > targ_idx:
                            continue
                    elif case == 'improvement':
                        ### slightly more complicated
                        check_idx = case_targets['ability'].index(targ.split('2')[1])
                        if pov_idx > check_idx:
                            continue
                for mode in modalities:
                    print('\n')
                    print(('target: ', targ,'pov: ', pov, mode))
                    if case == 'improvement':
                        ### for improvement, we remove variance from previous ability
                        additional_confound = '{}_score'.format(targ.split('2')[0])
                        #curr_confounds = confounds.copy() + [additional_confound + '{}_dps'.format(targ.split('2')[1])]
                        curr_confounds = confounds.copy() + [additional_confound]
                        targ_key = '{}_score'.format(targ.split('2')[1])
                    else:
                        targ_key = '{}_score'.format(targ)
                        ### for ability, we remove variance from dps
                        #curr_confounds = confounds.copy() + ['{}_dps'.format(targ)]
                        curr_confounds = confounds.copy()

                    if analysis_type == 'functional':
                        predictors = [d for d in dimensions if '_{}_'.format(pov) in d and mode in d]

                        curr_confounds = [d for d in dimensions if '_{}_'.format(pov) in d and mode not in d] + curr_confounds
                    else:
                        if targ == mode.replace('_score', ''):
                            continue
                        if 'score' in mode:
                            if case == 'ability':
                                if case_targets[case].index(targ) <= case_targets[case].index(mode.replace('_score', '')):
                                    continue
                            elif case == 'improvement':
                                if case_targets['ability'].index(targ.split('2')[1]) <= case_targets['ability'].index(mode.replace('_score', '')):
                                    continue
                                if case_targets['ability'].index(targ.split('2')[0]) == case_targets['ability'].index(mode.replace('_score', '')):
                                    continue
                        predictors = [d for d in dimensions if mode in d]
                        #curr_confounds = [c for c in curr_confounds if mode not in c and (('activity' in c and '_{}_'.format(targ) in c) or 'activity' not in c)]
                        new_predictors = list()
                        for p in predictors:
                            if 'score' in p:
                                if p != mode:
                                    continue
                            if mode not in p:
                                continue
                            new_predictors.append(p)
                        del predictors
                        predictors = list(set(new_predictors.copy()))

                        new_curr_confounds = list()
                        for c in curr_confounds:
                            if 'score' in mode and 'score' in c:
                                if case_targets['ability'].index(c.replace('_score', '')) >= case_targets['ability'].index(mode.replace('_score', '')):
                                    continue
                            if 'activity' in c or 'connectivity' in c:
                                if case == 'ability':
                                    if '_{}_'.format(targ) not in c:
                                        continue
                                if case == 'improvement':
                                    if '_{}_'.format(targ.split('2')[1]) not in c:
                                        continue
                            elif 'score' in c:
                                if mode == c:
                                    continue
                                if case == 'ability':
                                    if case_targets['ability'].index(c.replace('_score', '')) >= case_targets['ability'].index(targ):
                                        continue
                                if case == 'improvement':
                                    if case_targets['ability'].index(c.replace('_score', '')) >= case_targets['ability'].index(targ.split('2')[1]):
                                        continue
                            else:
                                if mode in c:
                                    continue
                            new_curr_confounds.append(c)
                        del curr_confounds
                        curr_confounds = list(set(new_curr_confounds.copy()))

                    print(['target: ', targ_key])
                    print(['predictors: ', sorted(predictors)])
                    print(['confounds: ', sorted(curr_confounds)])
                    print('\n')
                    assert len(predictors) in [1, 2, 4, 6, 8, 9]
                    if mode not in weights[case].keys():
                        weights[case][mode] = numpy.zeros(shape=(3, len(povs), len(predictors), iterations, folds))
                        weights_names[case][mode] = numpy.empty(shape=(3, len(povs), len(predictors)), dtype=numpy.dtypes.StringDType)
                    weights_names[case][mode][targ_idx, pov_idx, :] = predictors
                    for iteration in tqdm(range(iterations)):
                        for fold in range(folds):
                            train_items = sets['train'][fold]
                            assert len(train_items) == 47-test_size
                            test_items = sets['test'][fold]
                            assert len(test_items) == test_size
                            ### inputs
                            train_inputs = data[train_items, :][:, [dimensions.index(p) for p in predictors]]
                            assert train_inputs.shape[1] in [1, 2, 4, 6, 8, 9]
                            test_inputs = data[test_items, :][:, [dimensions.index(p) for p in predictors]]
                            assert test_inputs.shape[1] in [1, 2, 4, 6, 8, 9]
                            ### confounds
                            train_confounds = data[train_items, :][:, [dimensions.index(c) for c in curr_confounds]]
                            #assert train_confounds.shape[1] in [6, 7, 9, 12]
                            #print(train_confounds.shape)
                            assert train_confounds.shape[1] in [2, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15, 16, 17, 18, 19, 20, 21]
                            ### 11: x activity, 4x lesion, 1 age
                            test_confounds = data[test_items, :][:, [dimensions.index(c) for c in curr_confounds]]
                            #assert test_confounds.shape[1] in [6, 7, 9, 12]
                            assert test_confounds.shape[1] in [2, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15, 16, 17, 18, 19, 20, 21]
                            ### targets
                            train_targets = data[train_items, :][:, dimensions.index(targ_key)]
                            ### randomizing? for iterations>0
                            if iteration > 0:
                                train_targets = numpy.array(random.sample(train_targets.tolist(), k=len(train_targets)))
                            assert train_targets.shape == (47-test_size,)
                            test_targets = data[test_items, :][:, dimensions.index(targ_key)]
                            ### randomizing? for iterations>0
                            if iteration > 0:
                                test_targets = numpy.array(random.sample(test_targets.tolist(), k=len(test_targets)))
                            assert test_targets.shape == (test_size,)
                            ### residualization
                            model = linear_model.LinearRegression()
                            model.fit(train_confounds, train_targets)

                            train_toremove = model.predict(train_confounds)
                            test_toremove = model.predict(test_confounds)

                            train_targets_resid = train_targets-train_toremove
                            test_targets_resid = test_targets-test_toremove
                            ### training/testing
                            model = linear_model.RidgeCV(
                                                         alphas=alphas,
                                                         )
                            #model = linear_model.ElasticNetCV(alphas=alphas)
                            model.fit(train_inputs, train_targets_resid)
                            preds = model.predict(test_inputs)
                            ### evaluation
                            corr = scipy.stats.spearmanr(test_targets_resid, preds).statistic
                            ### weights
                            assert (model.coef_).shape == (len(predictors),)
                            weights[case][mode][targ_idx, pov_idx, :, iteration, fold] = model.coef_
                            results[case][mode][targ_idx, pov_idx, iteration, fold] = corr
                        #results[case][mode][targ_idx, pov_idx, iteration] = scipy.stats.trim_mean(iter_scores, 0.2, nan_policy='omit')
                    ### scores
                    avg = numpy.nanmean(results[case][mode][targ_idx, pov_idx, 0, :])
                    #p = (1+sum([1 for _ in results[case][mode][targ_idx, pov_idx, :] if _<0.]))/(iterations+1)
                    perm_avgs = numpy.nanmean(results[case][mode][targ_idx, pov_idx, 1:, :], axis=1)
                    t_maxes.append(perm_avgs.tolist())
                    assert perm_avgs.shape == (iterations-1, )
                    p = (sum([1 for v in perm_avgs if v>avg])+1)/iterations
                    print('average: {}, p-value (raw): {}\n'.format(round(avg, 3), round(p, 3)))
                    raw_ps.append((case, mode, targ_idx, pov_idx, p))

    ### one-tailed t max
    t_maxes = numpy.array(t_maxes)
    assert t_maxes.shape[1] == (iterations-1)
    t_maxes = numpy.max(t_maxes, axis=0)
    assert t_maxes.shape == (iterations-1,)
    print('t-max: {}'.format(round(numpy.percentile(t_maxes, 95), 4)))

    ### 0-> raw p, 1->fdr p
    ### correcting p-values for both ability and improvement at the same time
    ps = {m : {k : numpy.zeros(shape=(3, len(povs), 2)) for k in modalities} for m in ['ability', 'improvement']}
    fdr_ps = scipy.stats.false_discovery_control([v[-1] for v in raw_ps])
    print(fdr_ps)
    for _ in range(len(raw_ps)):
        case, mode, targ_idx, pov_idx, raw_p = raw_ps[_]
        fdr_p = fdr_ps[_]
        ps[case][mode][targ_idx, pov_idx, :] = [raw_p, fdr_p]
    all_results[analysis_type]['results'] = results
    all_results[analysis_type]['weights'] = weights
    all_results[analysis_type]['weights_names'] = weights_names
    all_results[analysis_type]['ps'] = ps
    all_results[analysis_type]['t-max'] = t_maxes

os.makedirs('pkls', exist_ok=True)

with open(os.path.join('pkls', 'full_results.pkl'), 'wb') as o:
    pickle.dump(all_results, o)

### variable by variable

var_by_var_results = dict()

for analysis_type in [
                      'functional',
                      'traditional',
                      ]:
    var_by_var_results[analysis_type] = {'details' : details[analysis_type]}
    modalities = details[analysis_type]['modalities']
    confounds = details[analysis_type]['confounds']
    povs = details[analysis_type]['povs']
    ### var by var model
    ### results for targets (always 3)
    ### max dimensionality is 9 (connectivity)
    removal_results = {m : {k : numpy.zeros(shape=(3, len(povs), 9, folds)) for k in modalities} for m in ['ability', 'improvement']}
    removal_predictors = {m : {k : numpy.empty(shape=(3, len(povs), 9), dtype=numpy.dtypes.StringDType) for k in modalities} for m in ['ability', 'improvement']}

    for case in results.keys():
        targets = case_targets[case]
        for targ_idx, targ in enumerate(targets):
            for pov_idx, pov in enumerate(povs):
                ### not interested in predicting the past from the future...
                if analysis_type == 'functional':
                    if case == 'ability':
                        if pov_idx > targ_idx:
                            continue
                    elif case == 'improvement':
                        ### slightly more complicated
                        check_idx = case_targets['ability'].index(targ.split('2')[1])
                        if pov_idx > check_idx:
                            continue
                for mode in modalities:
                    if mode == 'age' or 'score' in mode:
                        continue
                    print('\n')
                    print(('target: ', targ,'pov: ', pov, mode))
                    if case == 'improvement':
                        ### for improvement, we remove variance from previous ability
                        additional_confound = '{}_score'.format(targ.split('2')[0])
                        #curr_confounds = confounds.copy() + [additional_confound + '{}_dps'.format(targ.split('2')[1])]
                        curr_confounds = confounds.copy() + [additional_confound]
                        targ_key = '{}_score'.format(targ.split('2')[1])
                    else:
                        targ_key = '{}_score'.format(targ)
                        ### for ability, we remove variance from dps
                        #curr_confounds = confounds.copy() + ['{}_dps'.format(targ)]
                        curr_confounds = confounds.copy()

                    if analysis_type == 'functional':
                        predictors = [d for d in dimensions if '_{}_'.format(pov) in d and mode in d]

                        curr_confounds = [d for d in dimensions if '_{}_'.format(pov) in d and mode not in d] + curr_confounds
                    else:
                        if targ == mode.replace('_score', ''):
                            continue
                        if 'score' in mode:
                            if case == 'ability':
                                if case_targets[case].index(targ) <= case_targets[case].index(mode.replace('_score', '')):
                                    continue
                            elif case == 'improvement':
                                if case_targets['ability'].index(targ.split('2')[1]) <= case_targets['ability'].index(mode.replace('_score', '')):
                                    continue
                                if case_targets['ability'].index(targ.split('2')[0]) == case_targets['ability'].index(mode.replace('_score', '')):
                                    continue
                        predictors = [d for d in dimensions if mode in d]
                        #curr_confounds = [c for c in curr_confounds if mode not in c and (('activity' in c and '_{}_'.format(targ) in c) or 'activity' not in c)]
                        new_predictors = list()
                        for p in predictors:
                            if 'score' in p:
                                if p != mode:
                                    continue
                            if mode not in p:
                                continue
                            new_predictors.append(p)
                        del predictors
                        predictors = list(set(new_predictors.copy()))

                        new_curr_confounds = list()
                        for c in curr_confounds:
                            if 'score' in mode and 'score' in c:
                                if case_targets['ability'].index(c.replace('_score', '')) >= case_targets['ability'].index(mode.replace('_score', '')):
                                    continue
                            if 'activity' in c or 'connectivity' in c:
                                if case == 'ability':
                                    if '_{}_'.format(targ) not in c:
                                        continue
                                if case == 'improvement':
                                    if '_{}_'.format(targ.split('2')[1]) not in c:
                                        continue
                            elif 'score' in c:
                                if mode == c:
                                    continue
                                if case == 'ability':
                                    if case_targets['ability'].index(c.replace('_score', '')) >= case_targets['ability'].index(targ):
                                        continue
                                if case == 'improvement':
                                    if case_targets['ability'].index(c.replace('_score', '')) >= case_targets['ability'].index(targ.split('2')[1]):
                                        continue
                            else:
                                if mode in c:
                                    continue
                            new_curr_confounds.append(c)
                        del curr_confounds
                        curr_confounds = list(set(new_curr_confounds.copy()))

                    print(['target: ', targ_key])
                    print(['predictors: ', sorted(predictors)])
                    print(['confounds: ', sorted(curr_confounds)])
                    print('\n')
                    assert len(predictors) in [1, 2, 4, 6, 8, 9]
                    removal_predictors[case][mode][targ_idx, pov_idx, :len(predictors)] = predictors
                    ### here we switch iterations by predictors
                    for curr_pred_i, curr_pred in enumerate(predictors):
                        for fold in range(folds):
                            train_items = sets['train'][fold]
                            assert len(train_items) == 47-test_size
                            test_items = sets['test'][fold]
                            assert len(test_items) == test_size
                            ### inputs
                            train_inputs = data[train_items, :][:, [dimensions.index(p) for p in predictors if p!=curr_pred]]
                            assert train_inputs.shape[1] in [1, 3, 5, 8]
                            test_inputs = data[test_items, :][:, [dimensions.index(p) for p in predictors if p!=curr_pred]]
                            assert test_inputs.shape[1] in [1, 3, 5, 8]
                            ### confounds
                            train_confounds = data[train_items, :][:, [dimensions.index(c) for c in curr_confounds]]
                            #assert train_confounds.shape[1] in [6, 7, 9, 12]
                            #print(train_confounds.shape)
                            assert train_confounds.shape[1] in [2, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15, 16, 17, 18, 19, 20, 21]
                            ### 11: x activity, 4x lesion, 1 age
                            test_confounds = data[test_items, :][:, [dimensions.index(c) for c in curr_confounds]]
                            #assert test_confounds.shape[1] in [6, 7, 9, 12]
                            assert test_confounds.shape[1] in [2, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15, 16, 17, 18, 19, 20, 21]
                            ### targets
                            train_targets = data[train_items, :][:, dimensions.index(targ_key)]
                            assert train_targets.shape == (47-test_size,)
                            test_targets = data[test_items, :][:, dimensions.index(targ_key)]
                            assert test_targets.shape == (test_size,)
                            ### residualization
                            model = linear_model.LinearRegression()
                            model.fit(train_confounds, train_targets)

                            train_toremove = model.predict(train_confounds)
                            test_toremove = model.predict(test_confounds)

                            train_targets_resid = train_targets-train_toremove
                            test_targets_resid = test_targets-test_toremove
                            ### training/testing
                            model = linear_model.RidgeCV(
                                                         alphas=alphas,
                                                         )
                            #model = linear_model.ElasticNetCV(alphas=alphas)
                            model.fit(train_inputs, train_targets_resid)
                            preds = model.predict(test_inputs)
                            ### evaluation
                            corr = scipy.stats.spearmanr(test_targets_resid, preds).statistic
                            removal_results[case][mode][targ_idx, pov_idx, curr_pred_i, fold] = corr
                        ### scores
                        avg = numpy.nanmean(removal_results[case][mode][targ_idx, pov_idx, curr_pred_i, :])
                        full_avg = numpy.nanmean(all_results[analysis_type]['results'][case][mode][targ_idx, pov_idx, 0, :])
                        ### weights
                        w_names_full = all_results[analysis_type]['weights_names'][case][mode][targ_idx, pov_idx].tolist()
                        assert curr_pred in w_names_full
                        w_names_idx = w_names_full.index(curr_pred)
                        avg_w = numpy.nanmean(all_results[analysis_type]['weights'][case][mode][targ_idx, pov_idx, w_names_idx, 0, :])
                        perm_avg_w = numpy.nanmean(all_results[analysis_type]['weights'][case][mode][targ_idx, pov_idx, w_names_idx, 1:, :], axis=-1)
                        assert perm_avg_w.shape == (iterations-1, )
                        ### two-tailed p
                        p = min(1, ((sum([1 for _ in perm_avg_w if abs(_)>abs(avg_w)])+1)/iterations)*2)
                        print('removed predictor: {}, average: {}, decrease in performance: {}, raw p: {}\n'.format(curr_pred, round(avg, 3), round(avg-full_avg, 3), round(p, 3)))

    var_by_var_results[analysis_type]['results'] = removal_results
    var_by_var_results[analysis_type]['predictors'] = removal_predictors
    '''
                        raw_ps.append((case, mode, targ_idx, pov_idx, curr_pred_i, p))
    raw_ps = list()


    ### 0-> raw p, 1->fdr p
    ### correcting p-values for both ability and improvement at the same time
    ps = {m : {k : numpy.zeros(shape=(3, len(povs), 9, 2)) for k in modalities} for m in ['ability', 'improvement']}
    fdr_ps = scipy.stats.false_discovery_control([v[-1] for v in raw_ps])
    print(fdr_ps)
    for _ in range(len(raw_ps)):
        case, mode, targ_idx, pov_idx, curr_pred_i, raw_p = raw_ps[_]
        fdr_p = fdr_ps[_]
        ps[case][mode][targ_idx, pov_idx, curr_pred_i, :] = [raw_p, fdr_p]
    var_by_var_results[analysis_type]['ps'] = ps
    '''

with open(os.path.join('pkls', 'var-by-var_results.pkl'), 'wb') as o:
    pickle.dump(var_by_var_results, o)
