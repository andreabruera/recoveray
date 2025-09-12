import matplotlib
import numpy
import os
import pickle
import random
import scipy

from matplotlib import font_manager, pyplot
from tqdm import tqdm

def two_samples_permutation(one, two):
    real = abs(numpy.nanmean(one)-numpy.nanmean(two))
    fakes = list()
    for _ in tqdm(range(1000)):
        fake = random.sample(one+two, k=len(one)+len(two))
        fake_one = fake[:len(one)]
        fake_two = fake[len(one):]
        fake_diff = abs(numpy.nanmean(fake_one)-numpy.nanmean(fake_two))
        fakes.append(fake_diff)
    p = (sum([1 for _ in fakes if _>real])+1)/1001
    return p

out_f = os.path.join('prediction_plots', 'SI')
os.makedirs(out_f, exist_ok=True)

# Using Helvetica as a font
font_folder = '../fonts/'
if os.path.exists(font_folder):
    font_dirs = [font_folder, ]
    font_files = font_manager.findSystemFonts(fontpaths=font_dirs)
    for p in font_files:
        font_manager.fontManager.addfont(p)
    matplotlib.rcParams['font.family'] = 'Helvetica LT Std'

with open('ridge_predictions_si.pkl', 'rb') as i:
    all_res = pickle.load(i)
pov = 'all'

### abilities

for f_name, key in [
                    ('all_predictors', 'correlations'),
                    ('only_lesions', 'only lesions'),
                    ('only_age', 'only age'),
                    ]:

    target_positions = {
                 'T1' : 0,
                 'T2' : 1,
                 'T3' : 2,
                 }
    pov_positions = {
                 'all' : 0,
                 }

    n_folds = 50
    iter_null_hyp = 1000

    abilities = ['T1', 'T2', 'T3']
    xticks = ['Acute','Subacute','Chronic']
    if 'les' in f_name:
        case = 'lesions'
    elif 'age' in f_name:
        case = 'age'
    else:
        case = 'Age & lesions'
    legend_mapper = {
                     'all' : case.capitalize(),
                     }
    colors = {
                 'all' : 'plum',
                 }
    fig, ax = pyplot.subplots(figsize=(9, 10), constrained_layout=True)

    all_xs = list()
    all_ps = list()
    all_ys = list()
    curr_key = key
    pov_xs = list()
    pov_ys = list()
    pov_ps = list()
    scatters = list()
    for target in abilities:
        try:
            curr_corrs = all_res['abilities'][pov][target][curr_key]
        except KeyError:
            continue
        ### p value
        p_container = numpy.average(curr_corrs, axis=1)
        p_one = sum([1 for _ in p_container if _<=0])/iter_null_hyp
        p_two = sum([1 for _ in p_container if _>=0])/iter_null_hyp
        p = min(p_one, p_two)*2
        if p == 0:
            p = 1/1001
        scatters.append(p_container)
        assert p_container.shape == (iter_null_hyp, )
        pov_ps.append(p)
        ### aggregate result
        t_avg = numpy.nanmean(p_container)
        pov_ys.append(t_avg)
        pov_xs.append(target_positions[target]+pov_positions[pov])
    all_ys.append(pov_ys)
    all_xs.append(pov_xs)
    all_ps.append(pov_ps)
    #xs = [target_positions[a]+pov_positions[pov] for a in abilities]
    ax.plot(pov_xs, pov_ys, color=colors[pov], linewidth=5,)
    ax.bar(pov_xs, pov_ys, width=0.3,alpha=0.4,  color=colors[pov])
    #ax.errorbar(pov_xs, pov_ys, yerr=numpy.nanstd(p_container), color='gray', capsize=2)
    ax.scatter(pov_xs, pov_ys, s=300,label=legend_mapper[pov], marker='D', color=colors[pov], edgecolors='white', linewidth=2.,zorder=3)
    for x, scats in zip(pov_xs, scatters):
        ax.scatter([x+random.sample([corr*0.01 for corr in range(-10, 10)], k=1)[0] for _ in range(iter_null_hyp)],
                    scats,
                   s=50,
                   color=colors[pov],
                   alpha=0.05,
                   )
### correcting p values
    all_ps_flat = [p for _ in all_ps for p in _]
    all_xs_flat = [x for _ in all_xs for x in _]
    all_ys_flat = [numpy.nanmean(y) for _ in all_ys for y in _]
    corr_ps = scipy.stats.false_discovery_control(all_ps_flat)
    for x, p, y_avg in zip(all_xs_flat, corr_ps, all_ys_flat):
        if p < 0.05:
            #print([p, y_avg])
            if y_avg > 0.:
                p_y = 0.02
            else:
                p_y = -0.02
            ax.scatter(x, p_y, s=300, marker='*', color='black', edgecolors='white', linewidth=1.)

    ax.legend(
              loc=9,
              fontsize=16,
              ncols=3,
              handletextpad=0.,
              borderpad=0.3,
              columnspacing=1.
              )
    ax.hlines(y=[_*0.1 for _ in range(-3, 7)], xmin=-.5, xmax=2.5, linestyle='--',color='gray', alpha=0.2)
    ax.hlines(y=0., xmin=-.5, xmax=2.5, color='gray')
    ax.set_ylim(bottom=-.35, top=0.65)
    ax.text(s='Language ability', x=1, y=-.3, fontsize=25, fontweight='bold', fontstyle='italic', va='center', ha='center')
    pyplot.ylabel('Spearman correlation', fontsize=20, fontweight='bold')
    pyplot.xticks(ticks=[0, 1, 2], labels=xticks, fontsize=23, fontweight='bold')
    pyplot.yticks(fontsize=15)
    pyplot.savefig(os.path.join(out_f, 'SI_abilities_{}.jpg'.format(f_name)), dpi=300)
    pyplot.clf()
    pyplot.close()

### improvements

for f_name, key in [
                    ('all_predictors', 'correlations'),
                    ('only_lesions', 'only lesions'),
                    ('only_age', 'only age'),
                    ]:
    target_positions = {
                 'T2-T1' : 0,
                 'T3-T1' : 1,
                 'T3-T2' : 2,
                 }
    pov_positions = {
                 'all' : 0,
                 }

    n_folds = 50
    iter_null_hyp = 1000

    improvements = ['T2-T1', 'T3-T1', 'T3-T2']
    xticks = ['Early','Long-term','Late']
    if 'les' in f_name:
        case = 'lesions'
    elif 'age' in f_name:
        case = 'age'
    else:
        case = 'Age & lesions'
    legend_mapper = {
                     'all' : case.capitalize(),
                     }
    colors = {
                 'all' : 'sandybrown',
                 }
    fig, ax = pyplot.subplots(figsize=(9, 10), constrained_layout=True)

    all_xs = list()
    all_ys = list()
    all_ps = list()
    curr_key = key
    pov_xs = list()
    pov_ys = list()
    pov_ps = list()
    scatters = list()
    for target in improvements:
        try:
            curr_corrs = all_res['improvements'][pov][target][curr_key]
        except KeyError:
            #print([pov, target])
            continue
        p_container = numpy.average(curr_corrs, axis=1)
        p_one = sum([1 for _ in p_container if _<0])/iter_null_hyp
        p_two = sum([1 for _ in p_container if _>0])/iter_null_hyp
        p = min(p_one, p_two)*2
        if p == 0:
            p = 1/1001
        scatters.append(p_container)
        assert p_container.shape == (iter_null_hyp, )
        pov_ps.append(p)
        ### aggregate result
        t_avg = numpy.nanmean(p_container)
        pov_ys.append(t_avg)
        pov_xs.append(target_positions[target]+pov_positions[pov])
    all_xs.append(pov_xs)
    all_ps.append(pov_ps)
    all_ys.append(pov_ys)
    #xs = [target_positions[a]+pov_positions[pov] for a in abilities]
    ax.plot(pov_xs, pov_ys, color=colors[pov], linewidth=5,)
    ax.bar(pov_xs, pov_ys, width=0.24,alpha=0.4,  color=colors[pov])
    #ax.errorbar(pov_xs, pov_ys, yerr=numpy.nanstd(p_container), color='gray', capsize=2)
    ax.scatter(pov_xs, pov_ys, marker='D', s=300,label=legend_mapper[pov], color=colors[pov], edgecolors='white', linewidth=2.,zorder=3)
    for x, scats in zip(pov_xs, scatters):
        ax.scatter([x+random.sample([corr*0.01 for corr in range(-10, 10)], k=1)[0] for _ in range(iter_null_hyp)],
                    scats,
                   s=50,
                   color=colors[pov],
                   alpha=0.05,
                   )
### correcting p values
    all_ps_flat = [p for _ in all_ps for p in _]
    all_xs_flat = [x for _ in all_xs for x in _]
    all_ys_flat = [numpy.nanmean(y) for _ in all_ys for y in _]
    corr_ps = scipy.stats.false_discovery_control(all_ps_flat)
    for x, p, y_avg in zip(all_xs_flat, corr_ps, all_ys_flat):
        if p < 0.05:
            #print([p, y_avg])
            if y_avg > 0.:
                p_y = 0.02
            else:
                p_y = -0.02
            ax.scatter(x, p_y, s=300, marker='*', color='black', edgecolors='white', linewidth=1.)

    ax.legend(
              loc=9,
              fontsize=16,
              ncols=3,
              handletextpad=0.,
              borderpad=0.3,
              columnspacing=1.
              )
    ax.hlines(y=[_*0.1 for _ in range(-3, 7)], xmin=-.5, xmax=2.5, linestyle='--',color='gray', alpha=0.2)
    ax.hlines(y=0., xmin=-.5, xmax=2.5, color='gray')
    ax.set_ylim(bottom=-.35, top=0.65)
    ax.text(s='Language improvement', x=1, y=-.3, fontsize=25, fontweight='bold', fontstyle='italic',va='center', ha='center')
    pyplot.ylabel('Spearman correlation', fontsize=20, fontweight='bold')
    pyplot.xticks(ticks=[0, 1, 2], labels=xticks, fontsize=23, fontweight='bold')
    pyplot.yticks(fontsize=15)
    pyplot.savefig(os.path.join(out_f, 'SI_improvements_{}.jpg'.format(f_name)), dpi=300)
    pyplot.clf()
    pyplot.close()

### both
xtimes = ['T1', 'T2', 'T3', 'T2-T1', 'T3-T1', 'T3-T2']
xticks = ['Acute', 'Subacute', 'Chronic', 'Early','Long-term','Late']
widths = {
          'T1' : .14,
          'T2' : .14,
          'T3' : .14,
          'T2-T1' : .14,
          'T3-T2' : .14,
          'T3-T1' : .14
          }
pred_fams = [
             'only age',
             'only lesions',
            ]
pred_colours = {
        'only age' : 'plum',
        'only lesions' : 'gray',
             }
legend_mapper = {
        'only age' : 'Age',
        'only lesions' : 'Lesions',
        }
cases = ['abilities', 'improvements']
povs = ['T1', 'T2', 'T3']
label_set = set()

fig, ax = pyplot.subplots(
                          constrained_layout=True,
                          figsize=(12, 10),
                          )
all_ps = list()
all_xs = list()
all_ys = list()
all_details = list()
for case in cases:
    for targ in xtimes:
        targ_ys = list()
        targ_scats = list()
        targ_ps = list()
        targ_details = list()
        for pred_fam in pred_fams:
            key = pred_fam
            try:
                curr_corrs = all_res[case][pov][targ][key]
            except KeyError:
                continue
            p_container = numpy.average(curr_corrs, axis=1)
            p_one = sum([1 for _ in p_container if _<0])/iter_null_hyp
            p_two = sum([1 for _ in p_container if _>0])/iter_null_hyp
            p = min(p_one, p_two)*2
            if p == 0:
                p = 1/1001
            targ_scats.append(p_container)
            assert p_container.shape == (iter_null_hyp, )
            targ_ps.append(p)
            ### aggregate result
            t_avg = numpy.nanmean(p_container)
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
            curr_key = details['pred_fam']
            ax.bar(x, y, width=0.2, alpha=0.6,  color=pred_colours[curr_key])
            if curr_key[5:] not in label_set:
                ax.scatter(x, y, s=100, label=legend_mapper[curr_key], marker='D', edgecolors='white', linewidth=2.,zorder=3, color=pred_colours[curr_key],)
                label_set.add(curr_key[5:])
            else:
                ax.scatter(x, y, s=100, edgecolors='white', marker='D', linewidth=2.,zorder=3, color=pred_colours[curr_key],)
            ax.scatter([x+random.sample([corr*0.0075 for corr in range(-10, 10)], k=1)[0] for _ in range(iter_null_hyp)],
                        scats,
                       s=50,
                       color=pred_colours[curr_key],
                       alpha=0.05,
                       edgecolors='white',
                       )
corr_ps = scipy.stats.false_discovery_control(all_ps)
for x, p, y in zip(all_xs, corr_ps, all_ys):
    if p < 0.05:
        if y > 0.:
            p_y = 0.02
        else:
            p_y = -0.02
        ax.scatter(x, p_y, s=300, marker='*', color='black', edgecolors='white', linewidth=1.)
ax.text(
        x=3.7,
        y=0.6175,
        s='p<0.05',
        fontsize=20,
        va='center',
        )
ax.scatter(
   3.65,
   0.6175,
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
          #x=[_+0.5 for _ in range(3)],
          #x=[_+2+0.5 for _ in range(len(xs))],
          x = [0.5, 1.45],
          ymin=-.25,
          ymax=.38,
          alpha=0.2,
          linestyles='-',
          color='black',
          )
ax.vlines(
          x=[2.6],
          #x=[_+2+0.5 for _ in range(len(xs))],
          ymin=-.32,
          ymax=.38,
          linestyles='dotted',
          color='black',
          linewidth=5,
          )
ax.vlines(
          #x=[_+0.5 for _ in range(3, 5)],
          #x=[_+2+0.5 for _ in range(len(xs))],
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
ax.hlines(y=[_*0.1 for _ in range(-3, 7)], xmin=-.2, xmax=5.45, linestyle='--',color='gray', alpha=0.2)
ax.hlines(y=0., xmin=-.2, xmax=5.45, color='gray')
ax.set_ylim(bottom=-.35, top=0.65)
ax.legend(
          loc=9,
          fontsize=18,
          ncols=6,
          handletextpad=0.,
          borderpad=0.3,
          markerscale=2.,
          columnspacing=1.
          )
pyplot.savefig(os.path.join(out_f, 'SI_all.jpg'.format(f_name)), dpi=300)
pyplot.clf()
pyplot.close()
with open(os.path.join(out_f, 'SI_all_stats.tsv'), 'w') as o:
    o.write('target_type\tpredictor\ttarget_time_point\tavg_correlation\tfdr_corrected_p_value\traw_p_value\n')
    for dets, corr_p, raw_p, y in zip(all_details, corr_ps, all_ps, all_ys):
        o.write('{}\t{}\t{}\t'.format(dets['case'], dets['key'], dets['target']))
        o.write('{}\t{}\t{}\n'.format(round(y, 5), round(corr_p, 5), round(raw_p, 5)))
