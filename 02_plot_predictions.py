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

def cis(scores):
    lower = numpy.quantile(scores, 0.25)
    upper = numpy.quantile(scores, 0.975)
    return (round(float(lower), 5), round(float(upper), 5))

# Using Helvetica as a font
font_folder = '../fonts/'
if os.path.exists(font_folder):
    font_dirs = [font_folder, ]
    font_files = font_manager.findSystemFonts(fontpaths=font_dirs)
    for p in font_files:
        font_manager.fontManager.addfont(p)
    matplotlib.rcParams['font.family'] = 'Helvetica LT Std'

with open(os.path.join('pkls', 'main_text_ridge_predictions.pkl'), 'rb') as i:
    all_res = pickle.load(i)

time_mapper = {
               'T1' : 'Acute',
               'T2' : 'Subacute',
               'T3' : 'Chronic',
               'T2-T1' : 'Early',
               'T3-T1' : 'Long-term',
               'T3-T2' : 'Late',
               }

### abilities & improvements

mapper = {
          'abilities' : 'ability',
          'improvements' : 'improvement'
          }

out_f = os.path.join('prediction_plots', 'main_text')
curr_out_f = os.path.join(out_f, 'summary_plots')
os.makedirs(curr_out_f, exist_ok=True)
n_folds = 50
iter_null_hyp = 10000

for general_case in ['abilities', 'improvements']:
    for key in [
                'both', 'activations', 'connectivity'
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
            xticks = [time_mapper[t] for t in targets]
            pov_positions = {
                     'T1' : -.31,
                     'T2' : 0,
                     'T3' : .31,
                     }
            colors = {
                     'T1' : 'teal',
                     'T2' : 'orange',
                     'T3' : 'plum',
                     }
            width = 0.3
        elif general_case == 'improvements':
            targets = ['T2-T1', 'T3-T1', 'T3-T2']
            pov_positions = {
                     'T1' : -.25,
                     'T2' : 0,
                     'T3' : .25,
                     }
            colors = {
                 'T1' : 'mediumseagreen',
                 'T2' : 'sandybrown',
                 'T3' : 'mediumpurple',
                 }
            width = 0.24
        xticks = [time_mapper[t] for t in targets]
        if key == 'both':
            case = 'activity\n& connectivity'
        else:
            case = key
        legend_mapper = {
                         'T1' : 'Acute {}'.format(case),
                         'T2' : 'Subacute {}'.format(case),
                         'T3' : 'Chronic {}'.format(case),
                         }
        fig, ax = pyplot.subplots(figsize=(9, 10), constrained_layout=True)
        txt = list()

        all_xs = list()
        all_ps = list()
        all_ys = list()
        all_cis = list()
        curr_povs = list()
        curr_targs = list()
        for pov in legend_mapper.keys():
            if 'act' in key or 'con' in key:
                curr_key = '{} {}'.format(key, pov)
            else:
                curr_key = key
            pov_xs = list()
            pov_ys = list()
            pov_ps = list()
            scatters = list()
            for target in targets:
                try:
                    curr_corrs = all_res[general_case][pov][target]['correlations'][curr_key]
                except KeyError:
                    continue
                ### p value
                p_container = numpy.average(curr_corrs, axis=1)
                p_one = sum([1 for _ in p_container if _<=0])/iter_null_hyp
                p_two = sum([1 for _ in p_container if _>=0])/iter_null_hyp
                p = min(p_one, p_two)*2
                if p == 0:
                    p = 1/(iter_null_hyp+1)
                scatters.append(p_container)
                assert p_container.shape == (iter_null_hyp, )
                pov_ps.append(p)
                ### aggregate result
                t_avg = numpy.nanmean(p_container)
                pov_ys.append(t_avg)
                all_cis.append(cis(p_container))
                curr_povs.append(pov)
                curr_targs.append(target)
                pov_xs.append(target_positions[target]+pov_positions[pov])
            all_ys.append(pov_ys)
            all_xs.append(pov_xs)
            all_ps.append(pov_ps)
            ax.plot(pov_xs, pov_ys, color=colors[pov], linewidth=5,)
            ax.bar(pov_xs, pov_ys, width=width,alpha=0.4,  color=colors[pov])
            ax.scatter(pov_xs, pov_ys, s=300,label=legend_mapper[pov], marker='D', color=colors[pov], edgecolors='white', linewidth=2.,zorder=3)
            for x, scats in zip(pov_xs, scatters):
                ax.scatter([x+random.sample([corr*0.01 for corr in range(-10, 10)], k=1)[0] for _ in range(1000)],
                            random.sample(scats.tolist(), k=1000),
                           s=50,
                           color=colors[pov],
                           alpha=0.05,
                           )
### correcting p values
        all_ps_flat = [p for _ in all_ps for p in _]
        all_xs_flat = [x for _ in all_xs for x in _]
        assert len(all_cis) == len(all_xs_flat)
        all_ys_flat = [numpy.nanmean(y) for _ in all_ys for y in _]
        corr_ps = scipy.stats.false_discovery_control(all_ps_flat)
        for x, p, y_avg, pov, targ, ci, raw_p in zip(all_xs_flat, corr_ps, all_ys_flat, curr_povs, curr_targs, all_cis, all_ps_flat):
            txt.append((pov, targ, y_avg, p, raw_p, ci))
            if p < 0.05:
                if y_avg > 0.:
                    p_y = 0.02
                else:
                    p_y = -0.02
                ax.scatter(x, p_y, s=300, marker='*', color='black', edgecolors='white', linewidth=1.)
            elif p < 0.11:
                if y_avg > 0.:
                    p_y = 0.02
                else:
                    p_y = -0.02
                ax.scatter(x, p_y, s=200, marker='P', color='black', edgecolors='white', linewidth=1.)

        ax.legend(
                  loc=9,
                  fontsize=16,
                  ncols=3,
                  handletextpad=0.,
                  borderpad=0.3,
                  columnspacing=1.
                  )
        ax.hlines(y=[_*0.1 for _ in range(-3, 5)], xmin=-.5, xmax=2.5, linestyle='--',color='gray', alpha=0.2)
        ax.hlines(y=0., xmin=-.5, xmax=2.5, color='gray')
        ax.set_ylim(bottom=-.35, top=0.45)
        ax.text(s='Language {}'.format(mapper[general_case]), x=1, y=-.3, fontsize=25, fontweight='bold', fontstyle='italic', va='center', ha='center')
        pyplot.ylabel('Spearman correlation', fontsize=20, fontweight='bold')
        pyplot.xticks(ticks=[0, 1, 2], labels=xticks, fontsize=23, fontweight='bold')
        pyplot.yticks(fontsize=15)
        pyplot.savefig(os.path.join(curr_out_f, '{}_{}.jpg'.format(general_case, key)), dpi=300)
        with open(os.path.join(curr_out_f, '{}_{}.tsv'.format(general_case, key)), 'w') as o:
            o.write('target\tpredictor\taverage_spearman_rho\t95%_CI\tFDR_p\traw_p\n')
            for pov, targ, y_avg, p, raw_p, ci in txt:
                o.write('{} {}\t{} {}\t{}\t{}\t{}\t{}\n'.format(general_case, targ, key, pov, round(y_avg, 5), str(ci), round(p, 5), round(raw_p, 5)))
        pyplot.clf()
        pyplot.close()

### both
xtimes = ['T1', 'T2', 'T3', 'T2-T1', 'T3-T1', 'T3-T2']
xticks = [time_mapper[t] for t in xtimes]
widths = {
          'T1' : .14,
          'T2' : .26,
          'T3' : .38,
          'T2-T1' : .26,
          'T3-T2' : .38,
          'T3-T1' : .38
          }
pred_fams = [
             'activations',
             'connectivity',
            ]
pred_colours = {
        'activations T1' : 'chocolate',
        'activations T2' : 'sandybrown',
        'activations T3' : 'tan',
        'connectivity T1' :'teal',
        'connectivity T2' : 'mediumseagreen',
        'connectivity T3' : 'darkseagreen',
             }
legend_mapper = {
        'activations T1' : 'Acute activity',
        'activations T2' : 'Subacute activity',
        'activations T3' : 'Chronic activity',
        'connectivity T1' :'Acute connectivity',
        'connectivity T2' : 'Subacute connectivity',
        'connectivity T3' : 'Chronic connectivity',
        }
cases = ['abilities', 'improvements']
povs = ['T1', 'T2', 'T3']
label_set = set()

fig, ax = pyplot.subplots(
                          constrained_layout=True,
                          figsize=(20, 10),
                          )
txt = list()
all_ps = list()
all_xs = list()
all_ys = list()
all_cis = list()
curr_povs = list()
curr_fams = list()
curr_targs = list()
curr_cases = list()
all_details = list()
for case in cases:
    for targ in xtimes:
        targ_ys = list()
        targ_scats = list()
        targ_ps = list()
        targ_details = list()
        for pred_fam in pred_fams:
            for pov in povs:
                key = '{} {}'.format(pred_fam, pov)
                try:
                    curr_corrs = all_res[case][pov][targ]['correlations'][key]
                except KeyError:
                    continue
                curr_confs = all_res[case][pov][targ]['confounds'][key]
                check_confounds(case, targ, pred_fam, pov, curr_confs)
                curr_preds = all_res[case][pov][targ]['weights'][key].keys()
                check_preds(case, targ, pred_fam, pov, curr_preds)
                p_container = numpy.average(curr_corrs, axis=1)
                p_one = sum([1 for _ in p_container if _<0])/iter_null_hyp
                p_two = sum([1 for _ in p_container if _>0])/iter_null_hyp
                p = min(p_one, p_two)*2
                if p == 0:
                    p = 1/(iter_null_hyp+1)
                targ_scats.append(p_container)
                assert p_container.shape == (iter_null_hyp, )
                targ_ps.append(p)
                ### aggregate result
                t_avg = numpy.nanmean(p_container)
                all_cis.append(cis(p_container))
                curr_fams.append(pred_fam)
                curr_cases.append(case)
                curr_povs.append(pov)
                curr_targs.append(targ)
                targ_ys.append(t_avg)
                targ_details.append({'pov' : pov, 'pred_fam' : pred_fam, 'target': targ, 'case' : case, 'key' : key})
        all_details.extend(targ_details)
        all_ps.extend(targ_ps)
        all_ys.extend(targ_ys)
        base_x = xtimes.index(targ)
        if len(targ_ys) == 0:
            continue
        elif len(targ_ys) == 2:
            corrections = [corr for corr in numpy.linspace(-widths[targ], widths[targ], len(targ_ys))]
            missing_idxs = []
        else:
            corrections = [corr for corr in numpy.linspace(-widths[targ], widths[targ], len(targ_ys)+1)]
            if len(targ_ys) == 4:
                missing_idxs = [2]
            else:
                missing_idxs = [3]

        targ_xs = [base_x+corr for corr_i, corr in enumerate(corrections) if corr_i not in missing_idxs]
        assert len(targ_xs) == len(targ_ys)
        all_xs.extend(targ_xs)
        for i, x, y, scats, details in zip(range(len(targ_ys)), targ_xs, targ_ys, targ_scats, targ_details):
            curr_key = '{} {}'.format(details['pred_fam'], details['pov'])
            ax.bar(x, y, width=0.12, alpha=0.6,  color=pred_colours[curr_key])
            if curr_key[5:] not in label_set:
                ax.scatter(x, y, s=100, label=legend_mapper[curr_key], marker='D', edgecolors='white', linewidth=2.,zorder=3, color=pred_colours[curr_key],)
                label_set.add(curr_key[5:])
            else:
                ax.scatter(x, y, s=100, edgecolors='white', marker='D', linewidth=2.,zorder=3, color=pred_colours[curr_key],)
            ax.scatter([x+random.sample([corr*0.005 for corr in range(-10, 10)], k=1)[0] for _ in range(1000)],
                        random.sample(scats.tolist(), k=1000),
                       s=50,
                       color=pred_colours[curr_key],
                       alpha=0.05,
                       edgecolors='white',
                       )
corr_ps = scipy.stats.false_discovery_control(all_ps)
for x, p, y, ci, pov, targ, raw_p, fam, case in zip(all_xs, corr_ps, all_ys, all_cis, curr_povs, curr_targs, all_ps, curr_fams, curr_cases):
    txt.append((pov, targ, y, p, raw_p, ci, fam, case))
    if p < 0.05:
        if y > 0.:
            p_y = 0.02
        else:
            p_y = -0.02
        ax.scatter(x, p_y, s=300, marker='*', color='black', edgecolors='white', linewidth=1.)
    elif p < 0.11:
        if y > 0.:
            p_y = 0.02
        else:
            p_y = -0.02
        ax.scatter(x, p_y, s=200, marker='P', color='black', edgecolors='white', linewidth=1.)
ax.text(
        x=5.2,
        y=0.425,
        s='p<0.05',
        fontsize=20,
        va='center',
        )
ax.text(
        x=5.2,
        y=0.425,
        s='p<0.05',
        fontsize=20,
        va='center',
        )
ax.scatter(
   5.15,
   0.425,
   s=300,
   marker = '*',
    color='black',
    zorder=3.,
    edgecolors='white'
   )
ax.text(
        x=1,
        y=-0.3,
        s='Language ability',
        fontsize=30,
        ha='center',
        va='center',
        fontweight='bold'
        )
ax.text(
        x=4,
        y=-0.3,
        s='Language improvement',
        fontsize=30,
        ha='center',
        va='center',
        fontweight='bold'
        )
ax.vlines(
          x = [0.5, 1.45],
          ymin=-.25,
          ymax=.38,
          alpha=0.2,
          linestyles='-',
          color='black',
          )
ax.vlines(
          x=[2.6],
          ymin=-.32,
          ymax=.38,
          linestyles='dotted',
          color='black',
          linewidth=5,
          )
ax.vlines(
          x = [3.4, 4.5],
          ymin=-.25,
          ymax=.38,
          alpha=0.2,
          linestyles='-',
          color='black',
          )
ax.spines[['right', 'bottom', 'top']].set_visible(False)
ax.margins(x=.01, y=0.)
pyplot.ylabel('Spearman correlation', fontsize=20, fontweight='bold')
pyplot.xticks(ticks=[0, 1, 2, 3, 4, 5], labels=xticks, fontsize=23, fontweight='bold')
pyplot.yticks(fontsize=15)
ax.hlines(y=[_*0.1 for _ in range(-3, 5)], xmin=-.2, xmax=5.45, linestyle='--',color='gray', alpha=0.2)
ax.hlines(y=0., xmin=-.2, xmax=5.45, color='gray')
ax.set_ylim(bottom=-.35, top=0.45)
ax.legend(
          loc=9,
          fontsize=18,
          ncols=6,
          handletextpad=0.,
          borderpad=0.3,
          markerscale=2.,
          columnspacing=1.
          )
pyplot.savefig(os.path.join(curr_out_f, 'all.jpg'), dpi=300)
pyplot.clf()
pyplot.close()
with open(os.path.join(curr_out_f, 'all_stats.tsv'), 'w') as o:
    o.write('target\tpredictor\taverage_spearman_rho\t95%_CI\tFDR_p\traw_p\n')
    for pov, targ, y_avg, p, raw_p, ci, fam, case in txt:
        o.write('{} {}\t{} {}\t{}\t{}\t{}\t{}\n'.format(case, targ, fam, pov, round(y_avg, 5), str(ci), round(p, 5), round(raw_p, 5)))

### individual predictors
colours = {
        'activations' : {
            'L_SMA' : 'lightpink',
            'L_DLPFC' : 'mediumvioletred',
            'L_IFGorb' : 'paleturquoise',
            'L_PTL' : 'steelblue',
            'R_DLPFC' : 'darkseagreen',
            'R_IFGorb' : 'darkgreen',
           },
    'connectivity' : {
           'lDLPFC_to_lIFG' : 'palevioletred',
           'lDLPFC_to_lPTL'  : 'mediumorchid',
           'lSMA_to_lIFG' : 'hotpink',
           'lSMA_to_lPTL' : 'plum',
           'lIFG_to_lPTL' : 'cornflowerblue',
           'lPTL_to_lIFG' : 'mediumblue',
           'rIFG_to_lPTL' : 'silver',
           'rIFG_to_lIFG' : 'slategrey',
           'rDLPFC_to_lIFG' : 'yellowgreen',
           'rDLPFC_to_lPTL'  : 'mediumseagreen',
           }
           }

orders = {
        'activations' : [
            'L_SMA',
            'L_DLPFC',
            'L_IFGorb',
            'L_PTL',
            'R_DLPFC',
            'R_IFGorb',
            ],
    'connectivity' : [
           'lDLPFC_to_lIFG',
           'lDLPFC_to_lPTL',
           'lSMA_to_lIFG',
           'lSMA_to_lPTL',
           'lIFG_to_lPTL',
           'lPTL_to_lIFG',
           'rIFG_to_lIFG',
           'rDLPFC_to_lIFG',
           'rDLPFC_to_lPTL',
           ]
           }

possibilities = [
                 ('activations', 'T3', 'abilities', 'T3'),
                 ('connectivity', 'T3', 'abilities', 'T3'),
                 ('activations', 'T3', 'improvements', 'T3-T1'),
                 ('activations', 'T1', 'improvements', 'T3-T2'),
                 ('activations', 'T2', 'improvements', 'T3-T2'),
                 ('activations', 'T3', 'improvements', 'T3-T2'),
                 ('connectivity', 'T2', 'improvements', 'T2-T1'),
                 ('connectivity', 'T3', 'improvements', 'T3-T1'),
                 ('connectivity', 'T3', 'improvements', 'T3-T2'),
                 ]

### correcting p values
hard_ps = dict()
for general_case in ['correlations', 'weights']:
    if general_case == 'correlations':
        hard_ps['{}_chance'.format(general_case)] = list()
        hard_ps['{}_comparisons'.format(general_case)] = list()
    else:
        hard_ps[general_case] = list()
    for case in all_res.keys():
        for pov in all_res[case].keys():
            for targ in all_res[case][pov].keys():
                for fam in pred_fams:
                    p_check = (fam, pov, case, targ)
                    if p_check not in possibilities:
                        continue
                    if general_case == 'correlations':
                        all_ks = [k for k in all_res[case][pov][targ][general_case].keys() if fam in k and ' and ' in k]
                        assert len(all_ks) == len(orders[fam])
                        x_names = list()
                        for k in orders[fam]:
                            for a_k in all_ks:
                                if k in a_k:
                                    x_names.append(a_k)
                        assert len(x_names) == len(orders[fam])

                        basic_corrs = numpy.nanmean(all_res[case][pov][targ][general_case]['{} {}'.format(fam, pov)], axis=1)
                        basic_avg = numpy.nanmean(basic_corrs)
                    else:
                        all_ks = sorted([k for k in all_res[case][pov][targ][general_case]['{} {}'.format(fam, pov)].keys()])
                        assert len(all_ks) == len(orders[fam])
                        x_names = list()
                        for k in orders[fam]:
                            for a_k in all_ks:
                                if k in a_k:
                                    x_names.append(a_k)
                        assert len(x_names) == len(orders[fam])

                    for x_i, x in enumerate(x_names):
                        if general_case == 'correlations':
                            curr_corrs = all_res[case][pov][targ][general_case][x]
                        else:
                            curr_corrs = all_res[case][pov][targ][general_case]['{} {}'.format(fam, pov)][x]

                        p_container = numpy.average(curr_corrs, axis=1)
                        t_avg = numpy.nanmean(p_container)
                        if general_case == 'correlations':
                            ### p value
                            ### against chance
                            p = sum([1 for _ in p_container-basic_corrs if _>0])/iter_null_hyp
                            if p == 0:
                                p = 1/(iter_null_hyp+1)
                            key = (case, pov, targ, fam, x)
                            hard_ps['{}_chance'.format(general_case)].append((key, p))
                            ### against the other values
                            for x_i_two, x_two in enumerate(x_names):
                                if x_i_two <= x_i:
                                    continue
                                other_container = numpy.nanmean(all_res[case][pov][targ][general_case][x_two], axis=1)-basic_corrs
                                diff_container = p_container-basic_corrs
                                p_one = sum([1 for _ in diff_container-other_container if _<0])/iter_null_hyp
                                p_two = sum([1 for _ in diff_container-other_container if _>0])/iter_null_hyp
                                p = min(p_one, p_two)*2
                                if p == 0:
                                    p = 1/(iter_null_hyp+1)
                                key = (case, pov, targ, fam, tuple(sorted((x, x_two))))
                                hard_ps['{}_comparisons'.format(general_case)].append((key, p))
                        else:
                            p_one = sum([1 for _ in p_container if _<0])/iter_null_hyp
                            p_two = sum([1 for _ in p_container if _>0])/iter_null_hyp
                            p = min(p_one, p_two)*2
                            if p == 0:
                                p = 1/(iter_null_hyp+1)
                            key = (case, pov, targ, fam, x)
                            hard_ps[general_case].append((key, p))
hard_corr_ps = {k : list() for k in hard_ps.keys()}
for k, v in hard_ps.items():
    corr_ps = scipy.stats.false_discovery_control([c[1] for c in v])
    hard_corr_ps[k] = [(key[0], val) for key, val in zip(v, corr_ps)]

n_folds = 50
iter_null_hyp = 10000
pred_fams = ['activations', 'connectivity']
for general_case in [
                     'correlations',
                     #'weights',
                     ]:
    for case in all_res.keys():
        for pov in all_res[case].keys():
            for targ in all_res[case][pov].keys():
                for fam in pred_fams:
                    p_check = (fam, pov, case, targ)
                    if p_check not in possibilities:
                        continue
                    curr_out_f = os.path.join(out_f, 'individual_predictors', general_case, case, targ, pov,)
                    os.makedirs(curr_out_f, exist_ok=True)
                    if general_case == 'correlations':
                        all_ks = [k for k in all_res[case][pov][targ][general_case].keys() if fam in k and ' and ' in k]
                        assert len(all_ks) == len(orders[fam])
                        x_names = list()
                        for k in orders[fam]:
                            for a_k in all_ks:
                                if k in a_k:
                                    x_names.append(a_k)
                        assert len(x_names) == len(orders[fam])

                        basic_corrs = numpy.nanmean(all_res[case][pov][targ][general_case]['{} {}'.format(fam, pov)], axis=1)
                        basic_avg = numpy.nanmean(basic_corrs)
                    else:
                        all_ks = sorted([k for k in all_res[case][pov][targ][general_case]['{} {}'.format(fam, pov)].keys()])
                        assert len(all_ks) == len(orders[fam])
                        x_names = list()
                        for k in orders[fam]:
                            for a_k in all_ks:
                                if k in a_k:
                                    x_names.append(a_k)
                        assert len(x_names) == len(orders[fam])

                    fig, ax = pyplot.subplots(figsize=(9, 10), constrained_layout=True)

                    xs = list()
                    ys = list()
                    ps = list()
                    raw_ps = list()
                    curr_cis = list()
                    scats = list()
                    label_set = set()
                    p_keys = list()
                    p_avgs = list()
                    x_colors =list()
                    txt = list()
                    for x_i, x in enumerate(x_names):
                        ### retrieving pre-corrected p against chance
                        key = (case, pov, targ, fam, x)
                        weight_key = (case, pov, targ, fam, x.split(' ')[-1])
                        if general_case == 'correlations':
                            chance_p = [k[1] for k in hard_corr_ps['{}_chance'.format(general_case)] if k[0]==key]
                            raw_chance_p = [k[1] for k in hard_ps['{}_chance'.format(general_case)] if k[0]==key]
                            raw_weight_p = [k[1] for k in hard_ps['weights'] if k[0]==weight_key]
                            weight_p = [k[1] for k in hard_corr_ps['weights'] if k[0]==weight_key]
                            assert len(raw_weight_p) == 1
                            assert len(weight_p) == 1
                            weight_p = weight_p[0]
                        else:
                            weight_p = [k[1] for k in hard_corr_ps['weights'] if k[0]==key]
                            assert len(weight_p) == 1
                            weight_p = weight_p[0]
                            raw_weight_p = [k[1] for k in hard_ps[general_case] if k[0]==key]
                        assert len(chance_p) == 1
                        raw_chance_p = raw_weight_p[0]
                        ### we consider only the p for weights
                        ps.append(weight_p)
                        raw_ps.append(raw_chance_p)
                        if general_case == 'correlations':
                            curr_corrs = all_res[case][pov][targ][general_case][x]
                        else:
                            curr_corrs = all_res[case][pov][targ][general_case]['{} {}'.format(fam, pov)][x]

                        ### p value
                        p_container = numpy.average(curr_corrs, axis=1)
                        t_avg = numpy.nanmean(p_container)
                        if general_case == 'correlations':
                            ### plotting impact
                            scats.append(p_container-basic_corrs)
                            ### aggregate result
                            impact = t_avg-basic_avg
                            ys.append(impact)
                        else:
                            scats.append(p_container)
                            ys.append(t_avg)
                        assert p_container.shape == (iter_null_hyp, )
                        curr_cis.append(cis(p_container-basic_corrs))
                        if fam == 'activations':
                            if x_i > 1 and x_i < 4:
                                xs.append(x_i+1)
                            elif x_i >= 4:
                                xs.append(x_i+2)
                            else:
                                xs.append(x_i)
                        else:
                            if x_i >= 4 and x_i < 6:
                                xs.append(x_i+1)
                            elif x_i >= 6:
                                xs.append(x_i+2)
                            else:
                                xs.append(x_i)
                        color = colours[fam][orders[fam][x_i]]
                        x_colors.append(color)
                    for x, y, scat, x_name, color, p, raw_p, curr_ci in zip(xs, ys, scats, x_names, x_colors, ps, raw_ps, curr_cis):
                        weight_y = numpy.nanmean(numpy.nanmean(all_res[case][pov][targ]['weights']['{} {}'.format(fam, pov)][x_name.split(' ')[-1]], axis=1))
                        txt.append((pov, targ, y, p, raw_p, curr_ci, x_name, weight_y))
                        label = x_name.replace('lSMA', 'SMA').replace('L_SMA', 'SMA').split(' ')[-1][:-3].replace('_', ' ').strip().replace('orb', '').replace('l', 'left ').replace('r', 'right ').replace('L ', 'left ').replace('R ', 'right ').replace('PTleft', 'PTL')
                        ax.bar(
                                x,
                                y,
                               color = color,
                               alpha=0.3,
                                )
                        if label not in label_set:
                            label_set.add(label)
                            ax.scatter(
                                       x,
                                       y,
                                       color=color,
                                       s=300,
                                       label=label,
                                       marker='D',
                                       edgecolors='white',
                                       linewidth=2.,
                                       zorder=3,
                                       )
                        else:
                            ax.scatter(
                                       x,
                                       y,
                                       s=300,
                                       marker='D',
                                       edgecolors='white',
                                       linewidth=2.,
                                       zorder=3,
                                       color=color,
                                       )
                        if p < 0.05:
                            if y >= -0.01:
                                continue
                            if weight_y > 0.:
                                if fam == 'connectivity':
                                    p_x = x+0.4
                                else:
                                    p_x = x+0.275
                                p_y = 0.086
                                p_line = 0.1
                                marker = 6
                                color = 'red'
                                direction = 'left'
                                degrees = 315
                            else:
                                p_x = x+0.4
                                p_y = 0.114
                                p_line = 0.1
                                marker = 7
                                color = 'blue'
                                direction = 'right'
                                degrees = 45
                            rotation = Affine2D().rotate_deg(degrees)
                            ax.scatter(p_x,
                                       p_y-0.15+y,
                                       s=400,
                                       marker=MarkerStyle(marker, direction, rotation),
                                       color=color,
                                       linewidth=1.)
                            ax.scatter(x,
                                       p_line-0.15+y,
                                       s=400,
                                       marker=MarkerStyle('|', direction, rotation),
                                       color=color,
                                       linewidth=5.)
                    ax.legend(
                              loc=9,
                              fontsize=16,
                              ncols=3,
                              handletextpad=0.,
                              borderpad=0.3,
                              framealpha=1.,
                              columnspacing=1.
                              )
                    ax.hlines(y=0., xmin=-.5, xmax=2.5, color='gray')
                    if general_case == 'correlations':
                        pyplot.ylabel('Impact of individual predictor \nremoval on Spearman correlation', fontsize=20, fontweight='bold')
                        ax.set_ylim(bottom=.11, top=-.45)
                        ax.hlines(y=[_*0.1 for _ in range(-4, 6)], xmin=-.5, xmax=max(xs)+.5, linestyle='--',color='gray', alpha=0.2)
                        pyplot.yticks(ticks=[.1, 0., -.1, -.2, -.3], fontsize=15)
                    else:
                        pyplot.ylabel('Ridge regression weights', fontsize=20, fontweight='bold')
                        ax.set_ylim(bottom=-.1, top=.1)
                        ax.hlines(y=[_*0.01 for _ in range(-10, 10)], xmin=-.5, xmax=max(xs)+.5, linestyle='--',color='gray', alpha=0.2)
                    ax.spines[['right', 'bottom', 'top']].set_visible(False)
                    pyplot.xticks(ticks=[])
                    pyplot.savefig(os.path.join(curr_out_f, '{}_{}_{}_{}_{}.jpg'.format(general_case,fam, case, targ, pov)), dpi=300)
                    pyplot.clf()
                    pyplot.close()
                    if general_case == 'correlations':
                        txt_marker = 'impact_on_spearman_rho'
                    else:
                        txt_marker = 'regression_weights'
                    with open(os.path.join(curr_out_f, '{}_{}_{}_{}_{}.tsv'.format(general_case,fam, case, targ, pov)), 'w') as o:
                        o.write('target\tpredictor\t{}\tregression_weights\tFDR_p\traw_p\n'.format(txt_marker))
                        for pov, targ, y_avg, p, raw_p, ci, key, y_weight in txt:
                            o.write('{} {}\t{}\t{}\t{}\t{}\t{}\n'.format(case, targ, key.replace(' and ', ' w/o '), round(y_avg, 5), round(y_weight, 5), round(p, 5), round(raw_p, 5)))
