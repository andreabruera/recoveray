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

def rsa_encoding(train_input, train_target, test_input, test_target):
    ### prediction model
    preds = list()
    for ts, tg in zip(test_input, test_target):
        unweighted_pred = numpy.average([trg*-scipy.spatial.distance.euclidean(tr, ts) for tr, trg in zip(train_input, train_target)])
        denom_avg = numpy.average([-scipy.spatial.distance.euclidean(tr, ts) for tr, trg in zip(train_input, train_target)])
        norm_term = numpy.average([trg*denom_avg for trg in train_target])
        pred = unweighted_pred-norm_term
        preds.append(pred)
    return preds

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
abilities = [
             'T1',
             'T2',
             'T3',
             ]
### z-score aka 0 centering
#full_data = {k : (v-numpy.nanmean(v))/numpy.nanstd(v) if k not in abilities else v for k, v in raw_data.items() if len(v)>0}
### no z-score
full_data = {k : v for k, v in raw_data.items() if len(v)>0}

improvements = [
                'T2-T1',
                'T3-T2',
                'T3-T1',
                ]

age = [
       'Age',
       ]
lesions = [
           'L_DLPFC',
           'L_IFGorb',
           'L_PTL',
           'L_SMA',
           ]

### setting up data names
labels = dict()
labels['lesions'] = lesions
#assert len(labels['lesions']) == 3
assert len(labels['lesions']) == 4
labels['abilities'] = abilities
assert len(labels['abilities']) == 3
labels['improvements'] = improvements
assert len(labels['improvements']) == 3
labels['T1'] = ['T1']
labels['T2'] = ['T2']
labels['T3'] = ['T3']
labels['age'] = ['Age']
labels['activations T1'] = [k for k in full_data.keys() if \
                                     k not in abilities and \
                                     k not in improvements and \
                                     k not in lesions \
                                     and '_to_' not in k and \
                                     'T1' in k\
                                ]
assert len(labels['activations T1']) == 6
labels['activations T2'] = [k for k in full_data.keys() if \
                                          k not in abilities and \
                                          k not in improvements and \
                                          k not in lesions and \
                                          '_to_' not in k and \
                                          'T2' in k\
                                          ]
assert len(labels['activations T2']) == 6
labels['activations T3'] = [k for k in full_data.keys() if \
                                          k not in abilities and \
                                          k not in improvements and \
                                          k not in lesions and \
                                          '_to_' not in k and \
                                          'T3' in k\
                                          ]
assert len(labels['activations T3']) == 6
labels['connectivity T1'] = [k for k in full_data.keys() if \
                                          k not in abilities and \
                                          k not in improvements and \
                                          k not in lesions and \
                                          '_to_' in k and \
                                          'T1' in k\
                                          ]
assert len(labels['connectivity T1']) == 9
labels['connectivity T2'] = [k for k in full_data.keys() if \
                                           k not in abilities and \
                                           k not in improvements and \
                                           k not in lesions and \
                                           '_to_' in k and \
                                           'T2' in k\
                                           ]
assert len(labels['connectivity T2']) == 9
labels['connectivity T3'] = [k for k in full_data.keys() if \
                                           k not in abilities and \
                                           k not in improvements and \
                                           k not in lesions and \
                                           '_to_' in k and \
                                           'T3' in k\
                                           ]
assert len(labels['connectivity T3']) == 9
### predictors
predictors = [
             'age',
             'lesions',
              ]
confounds = [
              'activations T1',
              'activations T2',
              'activations T3',
              'connectivity T1',
              'connectivity T2',
              'connectivity T3',
             ]

n_folds = 50
iter_null_hyp = 1000
test_items = int(n_subjects*0.2)

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

    for t in targets:
        key, dict_t = read_dict_keys(name, t)
        if key not in collector.keys():
            collector[key] = {'all' : dict()}
        if dict_t not in collector[key]['all'].keys():
            collector[key]['all'][dict_t] = dict()
        ### full case
        confounds_names = [k for l in confounds for k in labels[l] if int(l[-1])==int(t[-1])]
        weights_names = [k for l in predictors for k in labels[l]]
        print('target: {}'.format(dict_t))
        print('\n')
        print('predictors: {}'.format(weights_names))
        print('confounds: {}'.format(confounds_names))
        it_predictors = numpy.array([full_data[k] for k in weights_names]).T
        it_confounds = numpy.array([full_data[k] for k in confounds_names]).T
        it_target = numpy.array(full_data[t])
        assert it_target.shape[0] == it_predictors.shape[0]
        iters = list()
        results_container = numpy.zeros(shape=(iter_null_hyp, n_folds))
        weights_container = numpy.zeros(shape=(iter_null_hyp, n_folds, it_predictors.shape[1]))
        for null in tqdm(range(iter_null_hyp)):
            subject_subsampling = random.choices(range(n_subjects), k=n_subjects)
            reals = list()
            fakes = list()
            all_preds = list()
            all_targs = list()
            for fold in range(n_folds):
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

                ### start
                ### not bootstrapping subjects
                #train_input = it_predictors[train_idxs]
                #confound_train_input = it_confounds[train_idxs]
                #train_target = it_target[train_idxs]
                #test_input = it_predictors[test_idxs]
                #confound_test_input = it_confounds[test_idxs]
                #test_target = it_target[test_idxs]
                ### end

                ### cv-confound residualization
                model = linear_model.LinearRegression()
                # residualizing target
                model.fit(confound_train_input, train_target)
                confound_train = model.predict(confound_train_input)
                train_target = train_target-confound_train
                confound_test = model.predict(confound_test_input)
                test_target = test_target-confound_test
                ### prediction model
                model = linear_model.RidgeCV(alphas=(1e-3, 1e-2, 1e-1, 1, 1e1, 1e2, 1e3, 1e4))
                #model = linear_model.LassoCV()
                model.fit(train_input, train_target)
                assert model.coef_.shape == (weights_container.shape[-1],)
                weights_container[null, fold] = model.coef_
                preds = model.predict(test_input)
                #preds = rsa_encoding(train_input, train_target, test_input, test_target)
                all_targs.append(test_target)
                all_preds.append(preds)
                corr = scipy.stats.spearmanr(test_target, preds).statistic
                #corr = scipy.stats.pearsonr(test_target, preds).statistic
                results_container[null, fold] = corr
                reals.append(corr)
            #r2 = sklearn.metrics.r2_score(all_targs, all_preds)
            #print(r2)
            t_avg = numpy.nanmean(reals)
            iters.append(t_avg)
        collector[key]['all'][dict_t]['correlations'] = results_container
        ### p value
        p_container = numpy.average(results_container, axis=1)
        assert p_container.shape == (iter_null_hyp, )
        p = (sum([1 for _ in p_container if _<0])+1)/(iter_null_hyp+1)
        #print(t_avg)
        ### aggregate result
        t_avg = numpy.nanmean(p_container)
        print([dict_t, 'avg: {}'.format(round(t_avg, 3)), 'p: {}'.format(round(p, 3))])
        ### weights
        collector[key]['all'][dict_t]['weights'] = weights_container
        collector[key]['all'][dict_t]['weights_names'] = weights_names
        #weights_z_mean = numpy.nanmean(weights_container, axis=2).reshape(weights_container.shape[0], weights_container.shape[1], 1)
        #weights_z_std = numpy.std(weights_container, axis=2).reshape(weights_container.shape[0], weights_container.shape[1], 1)
        #weights_z = (weights_container-weights_z_mean) / weights_z_std
        #assert weights_z.shape == weights_container.shape
        #avg_weights = numpy.nanmean(weights_z, axis=(0, 1))
        avg_weights = numpy.nanmean(weights_container, axis=(0, 1))
        assert avg_weights.shape == (it_predictors.shape[1],)
        assert (len(weights_names),) == avg_weights.shape
        avg_weights_dict = {k : v for k, v in zip(weights_names, avg_weights)}
        sorted_weights = sorted(avg_weights_dict.items(), key=lambda item : item[1], reverse=True)
        # weights p values
        weights_p = dict()
        for w_idx in range(weights_container.shape[-1]):
            current_weight = numpy.nanmean(weights_container[:, :, w_idx], axis=1)
            assert current_weight.shape == (iter_null_hyp, )
            weight_p_one = (sum([1 for _ in current_weight if _<0])+1)/(iter_null_hyp+1)
            weight_p_two = (sum([1 for _ in current_weight if _>0])+1)/(iter_null_hyp+1)
            weight_p = min([weight_p_one, weight_p_two])*2
            weights_p[weights_names[w_idx]] = weight_p
        #print(sorted(weights_p.items(), key=lambda item : item[1],))
        sorted_weights_ps = [(k, round(float(v), 3), round(weights_p[k], 5)) for k, v in sorted_weights]
        print(sorted_weights_ps)
        print('\n')

    ### removing families of predictors
    for pr in predictors:
        for t in targets:
            key, dict_t = read_dict_keys(name, t)
            if key not in collector.keys():
                collector[key] = {'all' : dict()}
            if dict_t not in collector[key]['all'].keys():
                collector[key]['all'][dict_t] = dict()
            ### the other predictor
            other_predictors = [l for l in predictors if l!=pr]
            assert len(other_predictors) == 1
            other_pr = other_predictors[0]
            ### full case
            orig_confounds_names = [k for l in confounds for k in labels[l] if int(l[-1])==int(t[-1])]
            other_confounds_names = [k for l in predictors for k in labels[l] if l==other_pr]
            confounds_names = orig_confounds_names+other_confounds_names
            weights_names = [k for l in predictors for k in labels[l] if l==pr]
            print('target: {} only {}'.format(dict_t, pr))
            print('\n')
            print('predictors: {}'.format(weights_names))
            print('confounds: {}'.format(confounds_names))
            ### not removing, random sampling with replacement
            it_predictors = numpy.array([full_data[k] for k in weights_names]).T
            it_confounds = numpy.array([full_data[k] for k in confounds_names]).T
            it_target = numpy.array(full_data[t])
            iters = list()
            results_container = numpy.zeros(shape=(iter_null_hyp, n_folds))
            for null in tqdm(range(iter_null_hyp)):
                #it_predictors = numpy.array([full_data[k] if l!=pr else random.sample(full_data[k].tolist(), k=len(full_data[k])) for l in predictors for k in labels[l]]).T
                assert it_target.shape[0] == it_predictors.shape[0]
                subject_subsampling = random.choices(range(n_subjects), k=n_subjects)
                reals = list()
                fakes = list()
                all_preds = list()
                all_targs = list()
                for fold in range(n_folds):
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
                    ### cv-confound residualization
                    model = linear_model.LinearRegression()
                    # residualizing target
                    model.fit(confound_train_input, train_target)
                    confound_train = model.predict(confound_train_input)
                    train_target = train_target-confound_train
                    confound_test = model.predict(confound_test_input)
                    test_target = test_target-confound_test
                    ### prediction model
                    model = linear_model.RidgeCV(alphas=(1e-3, 1e-2, 1e-1, 1, 1e1, 1e2, 1e3, 1e4))
                    #model = linear_model.LassoCV(eps=1e-5)
                    model.fit(train_input, train_target)
                    preds = model.predict(test_input)
                    #preds = rsa_encoding(train_input, train_target, test_input, test_target)
                    all_targs.append(test_target)
                    all_preds.append(preds)
                    corr = scipy.stats.spearmanr(test_target, preds).statistic
                    results_container[null, fold] = corr
                    reals.append(corr)
                #r2 = sklearn.metrics.r2_score(all_targs, all_preds)
                #print(r2)
                t_avg = numpy.nanmean(reals)
                iters.append(t_avg)
            collector[key]['all'][dict_t]['only {}'.format(pr)] = results_container
            ### p value
            p_container = numpy.average(results_container, axis=1)
            assert p_container.shape == (iter_null_hyp, )
            p = (sum([1 for _ in p_container if _<0])+1)/(iter_null_hyp+1)
            #print(t_avg)
            ### aggregate result
            t_avg = numpy.nanmean(p_container)
            print(['{} only {}'.format(dict_t, pr), 'avg: {}'.format(round(t_avg, 3)), 'p: {}'.format(round(p, 3))])
            print('\n')
    ### removing individual predictor among families of predictors
    for pr in predictors:
        if pr == 'age':
            continue
        for t in targets:
            key, dict_t = read_dict_keys(name, t)
            if key not in collector.keys():
                collector[key] = {'all' : dict()}
            if dict_t not in collector[key]['all'].keys():
                collector[key]['all'][dict_t] = dict()
            ### the other predictor
            other_predictors = [l for l in predictors if l!=pr]
            assert len(other_predictors) == 1
            other_pr = other_predictors[0]
            orig_confounds_names = [k for l in confounds for k in labels[l] if int(l[-1])==int(t[-1])]
            other_confounds_names = [k for l in predictors for k in labels[l] if l==other_pr]
            confounds_names = orig_confounds_names+other_confounds_names
            it_confounds = numpy.array([full_data[k] for k in confounds_names]).T
            for ind_pred in labels[pr]:
                ### full case
                weights_names = [k for l in predictors for k in labels[l] if l==pr and k!=ind_pred]
                print('target: {} only {} w/o {}'.format(dict_t, pr, ind_pred,))
                print('\n')
                print('predictors: {}'.format(weights_names))
                print('confounds: {}'.format(confounds_names))
                it_predictors = numpy.array([full_data[k] for k in weights_names]).T
                #it_confounds = numpy.array([full_data[k] for l in confounds for k in labels[l]]).T
                it_target = numpy.array(full_data[t])
                iters = list()
                results_container = numpy.zeros(shape=(iter_null_hyp, n_folds))
                for null in tqdm(range(iter_null_hyp)):
                    #it_predictors = numpy.array([full_data[k] if l!=pr else random.sample(full_data[k].tolist(), k=len(full_data[k])) for l in predictors for k in labels[l]]).T
                    assert it_target.shape[0] == it_predictors.shape[0]
                    subject_subsampling = random.choices(range(n_subjects), k=n_subjects)
                    reals = list()
                    fakes = list()
                    all_preds = list()
                    all_targs = list()
                    for fold in range(n_folds):
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
                        ### cv-confound residualization
                        model = linear_model.LinearRegression()
                        # residualizing target
                        model.fit(confound_train_input, train_target)
                        confound_train = model.predict(confound_train_input)
                        train_target = train_target-confound_train
                        confound_test = model.predict(confound_test_input)
                        test_target = test_target-confound_test
                        ### prediction model
                        model = linear_model.RidgeCV(alphas=(1e-3, 1e-2, 1e-1, 1, 1e1, 1e2, 1e3, 1e4))
                        #model = linear_model.LassoCV(eps=1e-5)
                        model.fit(train_input, train_target)
                        preds = model.predict(test_input)
                        #preds = rsa_encoding(train_input, train_target, test_input, test_target)
                        all_targs.append(test_target)
                        all_preds.append(preds)
                        corr = scipy.stats.spearmanr(test_target, preds).statistic
                        results_container[null, fold] = corr
                        reals.append(corr)
                    #r2 = sklearn.metrics.r2_score(all_targs, all_preds)
                    #print(r2)
                    t_avg = numpy.nanmean(reals)
                    iters.append(t_avg)
                collector[key]['all'][dict_t]['only {} and {}'.format(pr, ind_pred)] = results_container
                ### p value
                p_container = numpy.average(results_container, axis=1)
                assert p_container.shape == (iter_null_hyp, )
                p = (sum([1 for _ in p_container if _<0])+1)/(iter_null_hyp+1)
                #print(t_avg)
                ### aggregate result
                t_avg = numpy.nanmean(p_container)
                print(['{} only {} w/o {}'.format(dict_t, pr, ind_pred), 'avg: {}'.format(round(t_avg, 3)), 'p: {}'.format(round(p, 3))])
                print('\n')
with open('ridge_predictions_si.pkl', 'wb') as o:
    pickle.dump(collector, o)
