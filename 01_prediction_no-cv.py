import itertools
import matplotlib
import mne
import numpy
import os
import pickle
import random
import re
import scipy
import sklearn

from matplotlib import font_manager, pyplot
from scipy import spatial
from sklearn import linear_model, metrics, neural_network
from tqdm import tqdm

def read_dict_keys(name, t):
    if name == 'improvements':
        assert t in ['T3', 'T2']
        key = 'improvements'.format(t)
        dict_t = '{}-T1'.format(t)
    elif name == 'late improvements':
        assert t == 'T3'
        key = 'improvements'
        dict_t = '{}-T2'.format(t)
    else:
        key = 'abilities'
        dict_t = t
    return key, dict_t

def check_time_points(t, point_of_view, pr):
    check = True
    ### no pred from the future
    if int(point_of_view[-1]) > int(t[-1]):
        check = False
    ### only taking predictors from the current time point
    if int(pr[-1])!=int(point_of_view[-1]):
        check = False
    return check

def check_collector(collector, key, point_of_view, dict_t):
    if key not in collector.keys():
        collector[key] = {pov : dict() for pov in abilities}
    if point_of_view not in collector[key].keys():
        collector[key][point_of_view] = dict()
    if dict_t not in collector[key][point_of_view].keys():
        collector[key][point_of_view][dict_t] = {
                                                 'correlations' : dict(),
                                                 'weights' : dict(),
                                                 'confounds' : dict(),
                                                 }
    return collector

def generate_fold(n_subjects, test_items, it_predictors, it_target, it_confounds):
    test_idxs = random.sample(range(n_subjects), k=test_items)
    train_idxs = [k for k in range(n_subjects) if k not in test_idxs]
    ### bootstrapping subjects
    train_input = list()
    train_target = list()
    confound_train_input = list()
    for _ in train_idxs:
        train_input.append(it_predictors[_])
        train_target.append(it_target[_])
        confound_train_input.append(it_confounds[_])
    test_input = list()
    test_target = list()
    confound_test_input = list()
    for _ in test_idxs:
        test_input.append(it_predictors[_])
        test_target.append(it_target[_])
        confound_test_input.append(it_confounds[_])
    return train_input, train_target, confound_train_input, test_input, test_target, confound_test_input

def rsa_encoding(train_input, train_target, test_input, test_target):
    ### prediction model
    preds = list()
    coefs = numpy.zeros(shape=(len(train_input),))
    for ts, tg in zip(test_input, test_target):
        unweighted_pred = numpy.average([trg*-scipy.spatial.distance.euclidean(tr, ts) for tr, trg in zip(train_input, train_target)])
        denom_avg = numpy.average([-scipy.spatial.distance.euclidean(tr, ts) for tr, trg in zip(train_input, train_target)])
        norm_term = numpy.average([trg*denom_avg for trg in train_target])
        pred = unweighted_pred-norm_term
        preds.append(pred)

    return preds, coefs

def ridge_model(train_input, train_target, test_input, weights_names):
    ### prediction model
    model = linear_model.RidgeCV(alphas=(1e-3, 1e-2, 1e-1, 1, 1e1, 1e2, 1e3, 1e4))
    #model = linear_model.LassoCV()
    model.fit(train_input, train_target)
    preds = model.predict(test_input)

    return preds, model.coef_

def cv_confound(train_input, confound_train_input, train_target, test_input, confound_test_input, test_target):
    ### cv-confound residualization
    # we remove confounds from targets based on training set
    model = linear_model.LinearRegression()

    '''
    ### inputs
    model.fit(confound_train_input, train_input)

    confound_train = model.predict(confound_train_input)
    confound_test = model.predict(confound_test_input)

    train_input = train_input-confound_train
    test_input = test_input-confound_test
    '''

    ### targets
    #model.fit(confound_train_input, train_target)

    #confound_train = model.predict(confound_train_input)
    #confound_test = model.predict(confound_test_input)

    #train_target = train_target-confound_train
    #test_target = test_target-confound_test

    return train_input, train_target, test_input, test_target

def one_sample_bootstrap_p_value(one, side='both'):
    p_one = (sum([1 for _ in one if _<0]))/(iter_null_hyp)
    p_two = (sum([1 for _ in one if _>0]))/(iter_null_hyp)
    ### different hypotheses
    if side == 'both':
        p = min([p_one, p_two])*2
    elif side == 'lower':
        p = p_two
    elif side == 'greater':
        p = p_one
    else:
        raise RuntimeError('wrong direction')
    ### correcting if p==0
    if p == 0:
        p = 1/(iter_null_hyp+1)
    return p

def visual_check(results_container, weights_container, iter_null_hyp, point_of_view, dict_t):
    ### just for visual check while running code
    p_container = numpy.average(results_container, axis=1)
    #p_container = scipy.stats.trim_mean(results_container, 0.2, axis=1)
    assert p_container.shape == (iter_null_hyp, )
    p = (sum([1 for _ in p_container if _<0])+1)/(iter_null_hyp+1)
    #print(t_avg)
    ### aggregate result
    t_avg = numpy.nanmean(p_container)
    #t_avg = scipy.stats.trim_mean(p_container, 0.2)
    print(['{} from {}'.format(dict_t, point_of_view), 'avg: {}'.format(round(t_avg, 3)), 'p: {}'.format(round(p, 3))])
    ### weights
    avg_weights_dict = {k : numpy.nanmean(v) for k, v in weights_container.items()}
    sorted_weights = sorted(avg_weights_dict.items(), key=lambda item : item[1], reverse=True)
    # weights p values
    weights_p = dict()
    for w, v in weights_container.items():
        current_weight = numpy.nanmean(v, axis=1)
        assert current_weight.shape == (iter_null_hyp, )
        weight_p = one_sample_bootstrap_p_value(current_weight)
        weights_p[w] = weight_p
    sorted_weights_ps = [(k, round(float(v), 3), round(weights_p[k], 5)) for k, v in sorted_weights]
    print(sorted_weights_ps)
    print('\n')

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
raw_data['clinic'] = numpy.array(
                                 [1 for _ in range(14)] + \
                                 [2 for _ in range(14, 19)] + \
                                 [3 for _ in range(19, 37)] + \
                                 [4 for _ in range(37, 41)]
                                 )

### removing variance from targets
all_targets = numpy.zeros(shape=(6, 41))
for t in [1, 2, 3]:
    all_targets[t-1] = raw_data['T{}'.format(t)]
for t_i, t in enumerate([
               'T2-T1',
               'T3-T1',
               'T3-T2',
               ]):
    all_targets[t_i+3] = raw_data[t]
all_targets = all_targets.T

all_confounds = numpy.zeros(shape=(6, 41))
age = [
       'Age',
       ]
lesions = [
           'L_DLPFC',
           'L_IFGorb',
           'L_PTL',
           'L_SMA',
           ]
clinic = [
          'clinic',
          ]
#for c_i, c in enumerate(age):
#for c_i, c in enumerate(age+lesions+clinic):
for c_i, c in enumerate(age+lesions):
    all_confounds[c_i] = raw_data[c]
all_confounds = all_confounds.T
model = linear_model.LinearRegression()
model.fit(all_confounds, all_targets)
preds = model.predict(all_confounds)

resids = all_targets-preds

for t in [1, 2, 3]:
    raw_data['T{}'.format(t)] = resids[:, t-1]
for t_i, t in enumerate([
               'T2-T1',
               'T3-T1',
               'T3-T2',
               ]):
    raw_data[t] = resids[:, t_i+3]

### adding overall stuff
for t in [1, 2, 3]:
    overall_act = numpy.array([val for k, val in raw_data.items() if k[-1]==str(t) and len(k)>5 and '_to_' not in k], dtype=numpy.float32)
    overall_conn = numpy.array([val for k, val in raw_data.items() if k[-1]==str(t) and len(k)>5 and '_to_' in k], dtype=numpy.float32)
    assert overall_act.shape == (6, 41)
    assert overall_conn.shape == (9, 41)
    raw_data['avg_overall_act_T{}'.format(t)] = numpy.average(overall_act, axis=0)
    raw_data['std_overall_act_T{}'.format(t)] = numpy.std(overall_act, axis=0)
    raw_data['avg_overall_conn_T{}'.format(t)] = numpy.average(overall_conn, axis=0)
    raw_data['std_overall_conn_T{}'.format(t)] = numpy.std(overall_conn, axis=0)
raw_data['sub'] = numpy.array(list(range(1, n_subjects+1)))
assert l_i == n_subjects

abilities = [
             'T1',
             'T2',
             'T3',
             ]

full_data = {k : v for k, v in raw_data.items() if len(v)>0}
#full_data['overall_activations'] = numpy.average(overall_act, axis=0)
#full_data['overall_connectivity'] = numpy.average(overall_conn, axis=0)
#full_data['overall_activations_std'] = numpy.std(overall_act, axis=0)
#full_data['overall_connectivity_std'] = numpy.std(overall_conn, axis=0)

improvements = [
                'T2-T1',
                'T3-T2',
                'T3-T1',
                ]


### setting up data names
labels = dict()
labels['lesions'] = lesions
assert len(labels['lesions']) == 4
labels['abilities'] = abilities
assert len(labels['abilities']) == 3
labels['improvements'] = improvements
assert len(labels['improvements']) == 3
labels['T1'] = ['T1']
labels['T2'] = ['T2']
labels['T3'] = ['T3']
labels['age'] = ['Age']
labels['sub'] = ['sub']
labels['clinic'] = ['clinic']
labels['overall_act_T1'] = [
                            'avg_overall_act_T1',
                            'std_overall_act_T1',
                            ]
labels['overall_act_T2'] = [
                            'avg_overall_act_T2',
                            'std_overall_act_T2',
                            ]
labels['overall_act_T3'] = [
                            'avg_overall_act_T3',
                            'std_overall_act_T3',
                            ]
labels['overall_conn_T1'] = [
                            'avg_overall_conn_T1',
                            'std_overall_conn_T1',
                            ]
labels['overall_conn_T2'] = [
                            'avg_overall_conn_T2',
                            'std_overall_conn_T2',
                            ]
labels['overall_conn_T3'] = [
                            'avg_overall_conn_T3',
                            'std_overall_conn_T3',
                            ]

labels['activations T1'] = [k for k in full_data.keys() if \
                                     k not in abilities and \
                                     k not in improvements and \
                                     k not in lesions and \
                                     'overall' not in k and \
                                     '_to_' not in k and \
                                     'T1' in k\
                                ]
assert len(labels['activations T1']) == 6
labels['activations T2'] = [k for k in full_data.keys() if \
                                          k not in abilities and \
                                          k not in improvements and \
                                          k not in lesions and \
                                         'overall' not in k and \
                                          '_to_' not in k and \
                                          'T2' in k\
                                          ]
assert len(labels['activations T2']) == 6
labels['activations T3'] = [k for k in full_data.keys() if \
                                          k not in abilities and \
                                          k not in improvements and \
                                          k not in lesions and \
                                         'overall' not in k and \
                                          '_to_' not in k and \
                                          'T3' in k\
                                          ]
assert len(labels['activations T3']) == 6
labels['connectivity T1'] = [k for k in full_data.keys() if \
                                          k not in abilities and \
                                          k not in improvements and \
                                          k not in lesions and \
                                         'overall' not in k and \
                                          '_to_' in k and \
                                          'T1' in k\
                                          ]
assert len(labels['connectivity T1']) == 9
labels['connectivity T2'] = [k for k in full_data.keys() if \
                                           k not in abilities and \
                                           k not in improvements and \
                                           k not in lesions and \
                                         'overall' not in k and \
                                           '_to_' in k and \
                                           'T2' in k\
                                           ]
assert len(labels['connectivity T2']) == 9
labels['connectivity T3'] = [k for k in full_data.keys() if \
                                           k not in abilities and \
                                           k not in improvements and \
                                           k not in lesions and \
                                         'overall' not in k and \
                                           '_to_' in k and \
                                           'T3' in k\
                                           ]
assert len(labels['connectivity T3']) == 9
### predictors
predictors = [
              'activations T1',
              'activations T2',
              'activations T3',
              'connectivity T1',
              'connectivity T2',
              'connectivity T3',
              ]

data_across_t = numpy.zeros(shape=(
                                   41,
                                   len(labels['activations T1']) + \
                                   len(labels['activations T2']) + \
                                   len(labels['activations T3']) + \
                                   len(labels['connectivity T1']) + \
                                   len(labels['connectivity T2']) + \
                                   len(labels['connectivity T3'])
                                   )
                            )
for pr_i, pr in enumerate(predictors):
    if pr_i == 0:
        start = 0
    for l_i, l in enumerate(labels[pr]):
        data_across_t[:, start+l_i] = full_data[l].T
    start = start+l_i+1

test_items = int(n_subjects*0.2)

#for case, start, end in [
#                         ('activity', 0, 18),
#                         ('connectivity', 18, 41),
#                         ]:
print(data_across_t.shape)
for pr in predictors:
    mult = int(pr[-1])
    if 'act' in pr:
        start = 6*(mult-1)
        end = 6*(mult)
        assert start in range(0, 13)
        assert end in range(6, 19)
    if 'conn' in pr:
        start = 18+(9*(mult-1))
        end = 18+(9*mult)
        assert start in range(18, 37)
        assert end in range(27, 46)
    print((start, end))
    predictions = numpy.zeros(shape=(20, test_items, 6))
    reals = numpy.zeros(shape=(20, test_items, 6))
    for _ in range(20):
        iter_tests = random.sample(range(41), k=test_items)
        iter_trains = [_ for _ in range(41) if _ not in iter_tests]
        ### inputs
        train_input = data_across_t[iter_trains, start:end]
        test_input = data_across_t[iter_tests, start:end]
        ### targets
        train_target = all_targets[iter_trains, :]
        test_target = all_targets[iter_tests, :]
        ### confounds
        confound_train_input = all_confounds[iter_trains, :]
        confound_test_input = all_confounds[iter_tests, :]
        ### residualization
        resid_model = linear_model.LinearRegression()
        resid_model.fit(
                  confound_train_input,
                  train_target,
                  )
        resid_preds = resid_model.predict(
                                    confound_test_input
                                    )
        resids_test = test_target-resid_preds
        resid_preds = resid_model.predict(
                                    confound_train_input
                                    )
        resids_train = train_target-resid_preds

        model = linear_model.RidgeCV()
        model.fit(
                train_input,
                resids_train
                )
        preds = model.predict(
                              test_input
                              )
        assert preds.shape == (test_items, 6)
        predictions[_] = preds
        reals[_] = all_targets[iter_tests, :]

    corrs = numpy.zeros(shape=(20, 6))
    for _ in range(20):
        for __ in range(6):
            corrs[_, __] = scipy.stats.spearmanr(predictions[_, :, __], reals[_, :, __]).statistic
    print([pr, numpy.average(corrs, axis=0).tolist()])
import pdb; pdb.set_trace()




assert len(predictors) == 6
confounds = [
            'age',
            'lesions',
            #'clinic',
            #'sub',
            #'overall_activations',
            #'overall_connectivity',
            #'overall_activations_std',
            #'overall_connectivity_std',
             ]
assert len(confounds) == 2

n_folds = 50
iter_null_hyp = 599
test_items = int(n_subjects*0.2)
pred_model = 'ridge'

collector = dict()
for name, targets in [
                      ('abilities', abilities),
                      ('improvements', abilities[1:]),
                      ('late improvements', ['T3']),
                      ]:
    if name == 'improvements' and 'T1' not in confounds:
        print('\n\nimprovement over T1\n\n')
        if 'T1' not in confounds:
            confounds.append('T1')
    elif name == 'late improvements':
        print('\n\nlate improvement\n\n')
        if 'T1' in confounds:
            confounds = confounds[:-1]
            assert 'T1' not in confounds
        if 'T2' not in confounds:
            confounds.append('T2')
    else:
        print('\n\nlanguage ability\n\n')


    '''
    ### both
    last_key = 'both'
    for t in targets:
        key, dict_t = read_dict_keys(name, t)
        for point_of_view in abilities:

            ### no predictors from the future
            if int(point_of_view[-1]) > int(t[-1]):
                continue

            ### updating collector dict
            collector = check_collector(collector, key, point_of_view, dict_t)

            ### full case
            confounds_names = [k for l in confounds for k in labels[l]]
            ### predictors only from the present
            weights_names = [k for l in predictors for k in labels[l] if int(l[-1])==int(point_of_view[-1])]

            print('target: {} from {}'.format(dict_t, point_of_view))
            print('\n')
            print('predictors: {}'.format(weights_names))
            print('confounds: {}'.format(confounds_names))
            it_predictors = numpy.array([full_data[k] for k in weights_names]).T
            it_confounds = numpy.array([full_data[k] for k in confounds_names]).T
            it_target = numpy.array(full_data[t])
            assert it_target.shape[0] == it_predictors.shape[0]
            results_container = numpy.zeros(shape=(iter_null_hyp, n_folds))
            weights_container = {k : numpy.zeros(shape=(iter_null_hyp, n_folds)) for k in weights_names}
            for null in tqdm(range(iter_null_hyp)):
                subject_subsampling = random.choices(range(n_subjects), k=n_subjects)
                all_preds = list()
                all_targs = list()
                for fold in range(n_folds):
                    train_input, train_target, confound_train_input, \
                            test_input, test_target, confound_test_input = generate_fold(
                                                                                         n_subjects,
                                                                                         test_items,
                                                                                         it_predictors,
                                                                                         it_target,
                                                                                         it_confounds,
                                                                                         )

                    ### removing confounds
                    train_target, test_target = cv_confound(confound_train_input, train_target, confound_test_input, test_target)
                    ### training/predicting
                    if pred_model == 'ridge':
                        preds, coefs = ridge_model(train_input, train_target, test_input, weights_names)
                    elif pred_model == 'rsa_encoding':
                        preds, coefs = rsa_encoding(train_input, train_target, test_input, test_target)

                    ### storing

                    # weights
                    assert coefs.shape == (len(weights_container.keys()),)
                    for weight_i, weight in enumerate(weights_names):
                        weights_container[weight][null, fold] = coefs[weight_i]
                    # true values
                    all_targs.append(test_target)
                    # predictions
                    all_preds.append(preds)
                    # correlation
                    corr = scipy.stats.spearmanr(test_target, preds).statistic
                    results_container[null, fold] = corr

                ### r-squared
                #r2 = sklearn.metrics.r2_score(all_targs, all_preds)

            ### storing after iter_null_hyp*folds iterations
            collector[key][point_of_view][dict_t]['confounds'][last_key] = confounds_names
            collector[key][point_of_view][dict_t]['weights'][last_key] = weights_container
            collector[key][point_of_view][dict_t]['correlations'][last_key] = results_container

            ### just a visual check
            visual_check(results_container, weights_container, iter_null_hyp, point_of_view, dict_t)

            del results_container
            del weights_container
            del confounds_names
    '''

    ### removing families of predictors
    for pr in predictors:
        if 'conn' not in pr:
            continue
        for t in targets:
            if t != 'T3':
                continue
            '''
            ### additional predictors
            if 'conn' in pr:
                if '3' in t:
                    #confounds.append('overall_conn_T3')
                    #confounds.append('overall_conn_T2')
                    confounds.append('overall_conn_T1')
                else:
                    confounds.append('overall_conn_T2')
                    confounds.append('overall_conn_T1')
            elif 'act' in pr:
                if '3' in t:
                    confounds.append('overall_act_T3')
                    #confounds.append('overall_act_T2')
                    confounds.append('overall_act_T1')
                else:
                    #confounds.append('overall_act_T2')
                    confounds.append('overall_act_T1')
            '''
            key, dict_t = read_dict_keys(name, t)
            for point_of_view in abilities:
                if check_time_points(t, point_of_view, pr) == False:
                    continue
                ### updating collector dict
                collector = check_collector(collector, key, point_of_view, dict_t)
                ### the other predictor, from the present
                other_predictors = [l for l in predictors if l!=pr and int(l[-1])==int(point_of_view[-1])]
                assert len(other_predictors) == 1
                other_pr = other_predictors[0]

                ### confounds
                orig_confounds_names = [k for l in confounds for k in labels[l]]
                # predictors from the other family become confounds too
                other_confounds_names = [k for l in predictors for k in labels[l] if int(l[-1])==int(point_of_view[-1]) and l==other_pr]
                confounds_names = orig_confounds_names+other_confounds_names
                #confounds_names = orig_confounds_names

                ### weights from present & family
                weights_names = [k for l in predictors for k in labels[l] if int(l[-1])==int(point_of_view[-1]) and l==pr]

                ### just printing
                print('target: {} only {}, from {}'.format(dict_t, pr, point_of_view))
                print('\n')
                print('predictors: {}'.format(weights_names))
                print('confounds: {}'.format(confounds_names))

                it_predictors = numpy.array([full_data[k] for k in weights_names]).T
                it_confounds = numpy.array([full_data[k] for k in confounds_names]).T
                it_target = numpy.array(full_data[t])
                results_container = numpy.zeros(shape=(iter_null_hyp, n_folds))
                weights_container = {k : numpy.zeros(shape=(iter_null_hyp, n_folds)) for k in weights_names}
                for null in tqdm(range(iter_null_hyp)):
                    assert it_target.shape[0] == it_predictors.shape[0]
                    ### resampling with replacement for bootstrap
                    subject_subsampling = random.choices(range(n_subjects), k=n_subjects)
                    all_preds = list()
                    all_targs = list()
                    for fold in range(n_folds):
                        train_input, train_target, confound_train_input, \
                                test_input, test_target, confound_test_input = generate_fold(
                                                                                             n_subjects,
                                                                                             test_items,
                                                                                             it_predictors,
                                                                                             it_target,
                                                                                             it_confounds,
                                                                                             )

                        ### removing confounds
                        train_input, train_target, test_input, test_target = cv_confound(train_input, confound_train_input, train_target, test_input, confound_test_input, test_target)
                        ### training/predicting
                        if pred_model == 'ridge':
                            preds, coefs = ridge_model(train_input, train_target, test_input, weights_names)
                        elif pred_model == 'rsa_encoding':
                            preds, coefs = rsa_encoding(train_input, train_target, test_input, test_target)

                        ### storing

                        # weights
                        assert coefs.shape == (len(weights_container.keys()),)
                        for weight_i, weight in enumerate(weights_names):
                            weights_container[weight][null, fold] = coefs[weight_i]
                        # true values
                        all_targs.append(test_target)
                        # predictions
                        all_preds.append(preds)
                        # correlation
                        corr = scipy.stats.spearmanr(test_target, preds).statistic
                        results_container[null, fold] = corr

                    ### r-squared
                    #r2 = sklearn.metrics.r2_score(all_targs, all_preds)
                    #print(r2)

                ### storing after iter_null_hyp*folds iterations
                collector[key][point_of_view][dict_t]['confounds'][pr] = confounds_names
                collector[key][point_of_view][dict_t]['weights'][pr] = weights_container
                collector[key][point_of_view][dict_t]['correlations'][pr] = results_container

                ### just a visual check
                visual_check(results_container, weights_container, iter_null_hyp, point_of_view, dict_t)

                del results_container
                del weights_container
                del confounds_names


    '''
    ### removing individual predictor among families of predictors
    for pr in predictors:
        if 'conn' not in pr:
            continue
        for t in targets:
            key, dict_t = read_dict_keys(name, t)
            for point_of_view in abilities:
                if check_time_points(t, point_of_view, pr) == False:
                    continue
                ### updating collector dict
                collector = check_collector(collector, key, point_of_view, dict_t)
                ### the other predictor
                other_predictors = [l for l in predictors if l!=pr and int(l[-1])==int(point_of_view[-1])]
                assert len(other_predictors) == 1
                other_pr = other_predictors[0]
                for ind_pred in labels[pr]:

                    ### confounds
                    orig_confounds_names = [k for l in confounds for k in labels[l]]
                    # predictors from the other family become confounds too
                    other_confounds_names = [k for l in predictors for k in labels[l] if int(l[-1])==int(point_of_view[-1]) and l==other_pr]
                    confounds_names = orig_confounds_names+other_confounds_names

                    ### weights from present & family
                    weights_names = [k for l in predictors for k in labels[l] if int(l[-1])==int(point_of_view[-1]) and l==pr and k!=ind_pred]

                    ### just printing
                    print('target: {} only {} w/o {}, from {}'.format(dict_t, pr, ind_pred, point_of_view))
                    print('\n')
                    print('predictors: {}'.format(weights_names))
                    print('confounds: {}'.format(confounds_names))

                    it_confounds = numpy.array([full_data[k] for k in confounds_names]).T
                    it_predictors = numpy.array([full_data[k] for k in weights_names]).T
                    it_target = numpy.array(full_data[t])
                    results_container = numpy.zeros(shape=(iter_null_hyp, n_folds))
                    ### we don't really use the weight container, but we need it...
                    weights_container = {k : numpy.zeros(shape=(iter_null_hyp, n_folds)) for k in weights_names}
                    for null in tqdm(range(iter_null_hyp)):
                        assert it_target.shape[0] == it_predictors.shape[0]
                        subject_subsampling = random.choices(range(n_subjects), k=n_subjects)
                        all_preds = list()
                        all_targs = list()

                        for fold in range(n_folds):
                            train_input, train_target, confound_train_input, \
                                    test_input, test_target, confound_test_input = generate_fold(
                                                                                                 n_subjects,
                                                                                                 test_items,
                                                                                                 it_predictors,
                                                                                                 it_target,
                                                                                                 it_confounds,
                                                                                                 )

                            ### removing confounds
                            train_target, test_target = cv_confound(confound_train_input, train_target, confound_test_input, test_target)
                            ### training/predicting
                            if pred_model == 'ridge':
                                preds, coefs = ridge_model(train_input, train_target, test_input, weights_names)
                            elif pred_model == 'rsa_encoding':
                                preds, coefs = rsa_encoding(train_input, train_target, test_input, test_target)

                            ### storing

                            # here we don't do weights
                            # true values
                            all_targs.append(test_target)
                            # predictions
                            all_preds.append(preds)
                            # correlation
                            corr = scipy.stats.spearmanr(test_target, preds).statistic
                            results_container[null, fold] = corr

                        #r2 = sklearn.metrics.r2_score(all_targs, all_preds)
                        #print(r2)

                    ### storing after iter_null_hyp*folds iterations
                    last_key = '{} and {}'.format(pr, ind_pred)
                    collector[key][point_of_view][dict_t]['confounds'][last_key] = confounds_names
                    collector[key][point_of_view][dict_t]['correlations'][last_key] = results_container
                    ### weights are not saved here
                    try:
                        del collector[key][point_of_view][dict_t]['weights']
                    except KeyError:
                        continue

                    ### just a visual check
                    visual_check(results_container, weights_container, iter_null_hyp, point_of_view, dict_t)

                    del results_container
                    del weights_container
                    del confounds_names
    '''

out_f = 'pkls'
os.makedirs(out_f, exist_ok=True)
with open(os.path.join(out_f, 'main_text_{}_predictions_enriched.pkl'.format(pred_model)), 'wb') as o:
    pickle.dump(collector, o)
