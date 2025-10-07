import matplotlib
import numpy
import os
import pickle
import random
import scipy

from matplotlib import font_manager, pyplot
from matplotlib.transforms import Affine2D
from matplotlib.markers import MarkerStyle
from scipy import stats

def check_preds(case, targ, pred_fam, pov, curr_weights):
    ### checking everything is fine with the size
    if pred_fam == 'activations':
        pred_len = 6
    elif pred_fam == 'connectivity':
        pred_len = 9
    assert len(curr_weights) == pred_len
    ### checking everything is fine with time points
    for p in curr_weights:
        assert pov in p

def check_confounds(case, targ, pred_fam, pov, curr_confs):

    ### checking everything is fine with the size
    basic_conf_len = 5
    if pred_fam == 'activations':
        other_conf_len = 9
    elif pred_fam == 'connectivity':
        other_conf_len = 6
    if case == 'improvements':
        previous_abil = 1
    elif case == 'abilities':
        previous_abil = 0
    total_confs = basic_conf_len + other_conf_len + previous_abil
    assert len(curr_confs) == total_confs
    ### checking everything is fine with time points
    poss_ts = ['T1', 'T2', 'T3']
    for p in curr_confs:
        for t in poss_ts:
            if t==p:
                assert t == targ[-2:]
            elif t in p:
                assert t == pov

# Using Helvetica as a font
font_folder = '../fonts/'
if os.path.exists(font_folder):
    font_dirs = [font_folder, ]
    font_files = font_manager.findSystemFonts(fontpaths=font_dirs)
    for p in font_files:
        font_manager.fontManager.addfont(p)
    matplotlib.rcParams['font.family'] = 'Helvetica LT Std'

for general_case in ['abilities', 'improvements']:
    time_mapper = {
                   'T1' : 'Acute',
                   'T2' : 'Subacute',
                   'T3' : 'Chronic',
                   'T2-T1' : 'Early',
                   'T3-T1' : 'Long-term',
                   'T3-T2' : 'Late',
                   }
    fig, ax = pyplot.subplots(figsize=(20, 10), constrained_layout=True)

    ### abilities & improvements

    mapper = {
              'abilities' : 'ability',
              'improvements' : 'improvement'
              }

    curr_out_f = os.path.join('prediction_plots', 'lines')
    os.makedirs(curr_out_f, exist_ok=True)
    n_folds = 50
    iter_null_hyp = 10000
    povs = ['T1', 'T2', 'T3']
    comparisons = dict()
    comp_names = dict()
    for pred_fam, corr in [('main_text',-.1), ('si', .1)]:
        corr = 0.

        with open(os.path.join('pkls', '{}_ridge_predictions.pkl'.format(pred_fam)), 'rb') as i:
            all_res = pickle.load(i)

        if pred_fam == 'main_text':
            if general_case == 'abilities':
                color='orchid'
            else:
                color='darkorchid'
        else:
            if general_case == 'abilities':
                color='khaki'
            else:
                color='goldenrod'
        ys = dict()
        curr_preds = dict()
        for key in [
                    #'both',
                    'activations',
                    'connectivity',
                    'lesions',
                    'T1',
                    'T2',
                    'age',
                    ]:

            target_positions = {
                         'T1' : 0,
                         'T2' : 1,
                         'T3' : 2,
                         'T2-T1' : 0,
                         'T3-T1' : 1,
                         'T3-T2' : 2,
                         }
            if general_case == 'abilities':
                targets = ['T1', 'T2', 'T3']
            elif general_case == 'improvements':
                targets = ['T2-T1', 'T3-T1', 'T3-T2']
            xticks = [time_mapper[t] for t in targets]
            case = 'lines'

            for pov in povs:
                if 'act' in key or 'con' in key:
                    curr_key = '{} {}'.format(key, pov)
                else:
                    curr_key = key
                    pov = 'all'

                for target in targets:
                    try:
                        curr_corrs = all_res[general_case][pov][target]['correlations'][curr_key]
                    except KeyError:
                        continue
                    if target not in ys.keys():
                        ys[target] = list()
                        curr_preds[target] = list()
                    p_container = numpy.average(curr_corrs, axis=1)
                    #t_avg = numpy.nanmean(p_container)
                    ys[target].append(p_container)
                    curr_preds[target].append(curr_key)

        curr_xs = list()
        curr_ys = list()
        curr_errs = list()
        #for target, target_data in ys.items():
        for target in targets:
            target_data = [numpy.average(y) for y in ys[target]]
            x = target_positions[target]+corr
            curr_xs.append(x)
            #y = numpy.average(target_data)
            y = max(target_data)
            pred = curr_preds[target][target_data.index(y)]
            try:
                comparisons[target].append(ys[target][target_data.index(y)])
                comp_names[target].append(pred)
            except KeyError:
                comparisons[target] = [ys[target][target_data.index(y)]]
                comp_names[target] = [pred]
            curr_ys.append(y)
            y_err = numpy.std(ys[target][target_data.index(y)])
            curr_errs.append((y+y_err, y-y_err))
        ax.plot(curr_xs, curr_ys, color=color, linewidth=15,)
        ax.fill_between(curr_xs, [y[0] for y in curr_errs], [y[1] for y in curr_errs], color=color, alpha=0.1,)
        ax.scatter(curr_xs, curr_ys, s=1000, marker='D', color=color, edgecolors='white', linewidth=5.,zorder=3, alpha=0.8)
    txt_file = list()
    for target in targets:
        to_be_comp = comparisons[target]
        assert len(comparisons[target]) == 2
        diff = to_be_comp[0]-to_be_comp[1]
        p_one = sum([1 for _ in diff if _>0.])/len(diff)
        p_two = sum([1 for _ in diff if _<0.])/len(diff)
        p = min([p_one, p_two])*2
        if p == 0.:
            p = 1 / (1+len(diff))
        print([target, round(p, 5)])
        one = comp_names[target][0]
        two = comp_names[target][1]
        one_avg = numpy.mean(to_be_comp[0])
        two_avg = numpy.mean(to_be_comp[1])
        one_std = numpy.std(to_be_comp[0])
        two_std = numpy.std(to_be_comp[1])
        txt_file.append((one, one_avg, one_std, two, two_avg, two_std, p))
        if p < 0.05:
            x = target_positions[target]
            y = 0.12
            ax.scatter(x, y, s=1000, marker='*', color='black')
    with open(os.path.join(curr_out_f, '{}_{}.tsv'.format(general_case, case)), 'w') as o:
        o.write('func_neuro_pred\tfunc_neuro_pred_avg\tfunc_neuro_pred_std\t')
        o.write('trad_pred\ttrad_pred_avg\ttrad_pred_std\t')
        o.write('raw_p\t\n')
        for l in txt_file:
            for v_i, v in enumerate(l):
                if v_i in [1, 2, 4, 5, 6]:
                    v = round(v, 4)
                o.write('{}\t'.format(v))
            o.write('\n')

    ax.hlines(y=[_*0.1 for _ in range(-3, 8)], xmin=-.5, xmax=2.5, linestyle='--',color='gray', alpha=0.2)
    ax.hlines(y=0., xmin=-.5, xmax=2.5, color='gray')
    ax.set_ylim(bottom=-.02, top=0.55)
    #ax.text(s='Language {}'.format(mapper[general_case]), x=1, y=-.3, fontsize=25, fontweight='bold', fontstyle='italic', va='center', ha='center')
    pyplot.ylabel('Spearman correlation (best predictor)', fontsize=37, fontweight='bold')
    pyplot.xticks(ticks=[0, 1, 2], labels=xticks, fontsize=45, fontweight='bold')
    pyplot.yticks(fontsize=32)
    ax.spines[['right', 'bottom', 'top']].set_visible(False)
    pyplot.savefig(os.path.join(curr_out_f, '{}_{}.jpg'.format(general_case, case)), dpi=300)
