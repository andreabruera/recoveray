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
    p = sum([1 for _ in fakes if _>real])/1000
    return p*2
out_f = os.path.join('prediction_plots', 'main_text')
os.makedirs(out_f, exist_ok=True)

# Using Helvetica as a font
font_folder = '../fonts/'
if os.path.exists(font_folder):
    font_dirs = [font_folder, ]
    font_files = font_manager.findSystemFonts(fontpaths=font_dirs)
    for p in font_files:
        font_manager.fontManager.addfont(p)
    matplotlib.rcParams['font.family'] = 'Helvetica LT Std'

with open('ridge_predictions.pkl', 'rb') as i:
    all_res = pickle.load(i)

### abilities

for f_name, key in [
                    ('all_predictors', 'correlations'),
                    ('only_activations', 'only activations '),
                    ('only_connectivity', 'only connectivity '),
                    ]:

    target_positions = {
                 'T1' : 0,
                 'T2' : 1,
                 'T3' : 2,
                 }
    pov_positions = {
                 'T1' : -.31,
                 'T2' : 0,
                 'T3' : .31,
                 }

    n_folds = 50
    iter_null_hyp = 1000

    abilities = ['T1', 'T2', 'T3']
    xticks = ['Acute','Subacute','Chronic']
    if 'act' in f_name:
        case = 'activity'
    elif 'conn' in f_name:
        case = 'connectivity'
    else:
        case = 'activity\n&connectivity'
    legend_mapper = {
                     'T1' : 'Acute {}'.format(case),
                     'T2' : 'Subacute {}'.format(case),
                     'T3' : 'Chronic {}'.format(case),
                     }
    colors = {
                 'T1' : 'teal',
                 'T2' : 'orange',
                 'T3' : 'plum',
                 }
    fig, ax = pyplot.subplots(figsize=(9, 10), constrained_layout=True)

    all_xs = list()
    all_ps = list()
    all_ys = list()
    for pov in abilities:
        if 'act' in f_name or 'con' in f_name:
            curr_key = '{}{}'.format(key, pov)
        else:
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
    ax.hlines(y=[_*0.1 for _ in range(-3, 5)], xmin=-.5, xmax=2.5, linestyle='--',color='gray', alpha=0.2)
    ax.hlines(y=0., xmin=-.5, xmax=2.5, color='gray')
    ax.set_ylim(bottom=-.35, top=0.45)
    ax.text(s='Language ability', x=1, y=-.3, fontsize=25, fontweight='bold', fontstyle='italic', va='center', ha='center')
    pyplot.ylabel('Spearman correlation', fontsize=20, fontweight='bold')
    pyplot.xticks(ticks=[0, 1, 2], labels=xticks, fontsize=23, fontweight='bold')
    pyplot.yticks(fontsize=15)
    pyplot.savefig(os.path.join(out_f, 'abilities_{}.jpg'.format(f_name)), dpi=300)
    pyplot.clf()
    pyplot.close()

### improvements

for f_name, key in [
                    ('all_predictors', 'correlations'),
                    ('only_activations', 'only activations '),
                    ('only_connectivity', 'only connectivity '),
                    ]:
    target_positions = {
                 'T2-T1' : 0,
                 'T3-T1' : 1,
                 'T3-T2' : 2,
                 }
    pov_positions = {
                 'T1' : -.25,
                 'T2' : 0,
                 'T3' : .25,
                 }

    n_folds = 50
    iter_null_hyp = 1000

    improvements = ['T2-T1', 'T3-T1', 'T3-T2']
    xticks = ['Early','Long-term','Late']
    if 'act' in f_name:
        case = 'activity'
    elif 'conn' in f_name:
        case = 'connectivity'
    else:
        case = 'activity\n& connectivity'
    legend_mapper = {
                     'T1' : 'Acute {}'.format(case),
                     'T2' : 'Subacute {}'.format(case),
                     'T3' : 'Chronic {}'.format(case),
                     }
    colors = {
                 'T1' : 'mediumseagreen',
                 'T2' : 'sandybrown',
                 'T3' : 'mediumpurple',
                 }
    fig, ax = pyplot.subplots(figsize=(9, 10), constrained_layout=True)

    all_xs = list()
    all_ys = list()
    all_ps = list()
    for pov in abilities:
        if 'act' in f_name or 'con' in f_name:
            curr_key = '{}{}'.format(key, pov)
        else:
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
            #print(p_container)
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
    ax.hlines(y=[_*0.1 for _ in range(-3, 5)], xmin=-.5, xmax=2.5, linestyle='--',color='gray', alpha=0.2)
    ax.hlines(y=0., xmin=-.5, xmax=2.5, color='gray')
    ax.set_ylim(bottom=-.35, top=0.45)
    ax.text(s='Language improvement', x=1, y=-.3, fontsize=25, fontweight='bold', fontstyle='italic',va='center', ha='center')
    pyplot.ylabel('Spearman correlation', fontsize=20, fontweight='bold')
    pyplot.xticks(ticks=[0, 1, 2], labels=xticks, fontsize=23, fontweight='bold')
    pyplot.yticks(fontsize=15)
    pyplot.savefig(os.path.join(out_f, 'improvements_{}.jpg'.format(f_name)), dpi=300)
    pyplot.clf()
    pyplot.close()

### both
xtimes = ['T1', 'T2', 'T3', 'T2-T1', 'T3-T1', 'T3-T2']
xticks = ['Acute', 'Subacute', 'Chronic', 'Early','Long-term','Late']
widths = {
          'T1' : .14,
          'T2' : .26,
          'T3' : .38,
          'T2-T1' : .26,
          'T3-T2' : .38,
          'T3-T1' : .38
          }
pred_fams = [
             'only activations ',
             'only connectivity ',
            ]
pred_colours = {
        #'only activations T1' : 'teal',
        #'only activations T2' : 'mediumseagreen',
        #'only activations T3' : 'darkseagreen',
        'only activations T1' : 'chocolate',
        'only activations T2' : 'sandybrown',
        'only activations T3' : 'tan',
        'only connectivity T1' :'teal',
        'only connectivity T2' : 'mediumseagreen',
        'only connectivity T3' : 'darkseagreen',
        #'only connectivity T1' :'mediumpurple',
        #'only connectivity T2' : 'palevioletred',
        #'only connectivity T3' : 'plum',
             }
legend_mapper = {
        'only activations T1' : 'Acute activity',
        'only activations T2' : 'Subacute activity',
        'only activations T3' : 'Chronic activity',
        'only connectivity T1' :'Acute connectivity',
        'only connectivity T2' : 'Subacute connectivity',
        'only connectivity T3' : 'Chronic connectivity',
        }
cases = ['abilities', 'improvements']
povs = ['T1', 'T2', 'T3']
label_set = set()

fig, ax = pyplot.subplots(
                          constrained_layout=True,
                          figsize=(20, 10),
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
            for pov in povs:
                key = '{}{}'.format(pred_fam, pov)
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
            curr_key = '{}{}'.format(details['pred_fam'], details['pov'])
            ax.bar(x, y, width=0.12, alpha=0.6,  color=pred_colours[curr_key])
            if curr_key[5:] not in label_set:
                ax.scatter(x, y, s=100, label=legend_mapper[curr_key], marker='D', edgecolors='white', linewidth=2.,zorder=3, color=pred_colours[curr_key],)
                label_set.add(curr_key[5:])
            else:
                ax.scatter(x, y, s=100, edgecolors='white', marker='D', linewidth=2.,zorder=3, color=pred_colours[curr_key],)
            ax.scatter([x+random.sample([corr*0.005 for corr in range(-10, 10)], k=1)[0] for _ in range(iter_null_hyp)],
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
pyplot.savefig(os.path.join(out_f, 'all.jpg'.format(f_name)), dpi=300)
pyplot.clf()
pyplot.close()
with open(os.path.join(out_f, 'all_stats.tsv'), 'w') as o:
    o.write('target_type\tpredictor\ttarget_time_point\tavg_correlation\tfdr_corrected_p_value\traw_p_value\n')
    for dets, corr_p, raw_p, y in zip(all_details, corr_ps, all_ps, all_ys):
        o.write('{}\t{}\t{}\t'.format(dets['case'], dets['key'], dets['target']))
        o.write('{}\t{}\t{}\n'.format(round(y, 5), round(corr_p, 5), round(raw_p, 5)))

### individual predictors
colours = {
    'activations' : [
           'lightpink',
           'forestgreen',
           'steelblue',
           'khaki',
           'paleturquoise',
           'mediumvioletred',
           ],
    'connectivity' : [
           'paleturquoise',
           'mediumaquamarine',
           'forestgreen',
           'mediumblue',
           'palegoldenrod',
           'khaki',
           'palevioletred',
           'mediumorchid',
           'silver',
           ]
    }

n_folds = 50
iter_null_hyp = 1000
pred_fams = ['activations', 'connectivity']
for case in all_res.keys():
    for pov in all_res[case].keys():
        for targ in all_res[case][pov].keys():
            for fam in pred_fams:
                curr_out_f = os.path.join(out_f, 'individual_predictors', case, targ, pov,)
                os.makedirs(curr_out_f, exist_ok=True)
                x_names = sorted([k for k in all_res[case][pov][targ].keys() if fam in k and ' and ' in k])
                basic_corrs = numpy.nanmean(all_res[case][pov][targ]['only {} {}'.format(fam, pov)], axis=1)
                basic_avg = numpy.nanmean(basic_corrs)

                fig, ax = pyplot.subplots(figsize=(9, 10), constrained_layout=True)

                xs = list()
                ys = list()
                ps = list()
                scats = list()
                label_set = set()
                for x_i, x in enumerate(x_names):
                    curr_corrs = all_res[case][pov][targ][x]
                    ### p value
                    p_container = numpy.average(curr_corrs, axis=1)
                    #p_one = sum([1 for _ in p_container if _<=0])/iter_null_hyp
                    #p_two = sum([1 for _ in p_container if _>=0])/iter_null_hyp
                    #p_one = sum([1 for _,__ in zip(p_container, basic_corrs) if _<=__])/iter_null_hyp
                    #p = two_samples_permutation(p_container.tolist(), basic_corrs.tolist())
                    #p = scipy.stats.wilcoxon(p_container, basic_corrs).pvalue
                    #p = min(p_one, p_two)*2
                    ### one-tailed: in how many cases were correlations stronger for one as opposed to the other?
                    p = sum([1 for _,__ in zip(p_container, basic_corrs) if _>=__])/iter_null_hyp
                    if p == 0:
                        p = 1/1001
                    ### plotting impact
                    scats.append(p_container-basic_corrs)
                    assert p_container.shape == (iter_null_hyp, )
                    ps.append(p)
                    ### aggregate result
                    t_avg = numpy.nanmean(p_container)
                    impact = t_avg-basic_avg
                    ys.append(impact)
                    xs.append(x_i)
                    for x, y, scat, x_name in zip(xs, ys, scats, x_names):
                        label = x_name.split(' ')[-1][:-3].replace('_', ' ')
                        ax.bar(
                                x,
                                y,
                               color=colours[fam][x],
                               alpha=0.3,
                                )
                        if label not in label_set:
                            label_set.add(label)
                            ax.scatter(
                                       x,
                                       y,
                                       color=colours[fam][x],
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
                                       color=colours[fam][x],
                                       )
                        ax.scatter([x+random.sample([corr*0.03 for corr in range(-10, 10)], k=1)[0] for _ in range(iter_null_hyp)],
                                    scat,
                                   s=50,
                                   alpha=0.05,
                                    color=colours[fam][x],
                                   edgecolors='white'
                                   )
                ### correcting p values
                corr_ps = scipy.stats.false_discovery_control(ps)
                for x, p, y_avg in zip(xs, corr_ps, ys):
                    if p < 0.05:
                        #print([p, y_avg])
                        if y_avg > 0.:
                            p_y = 0.02
                        else:
                            p_y = -0.02
                        ax.scatter(
                                   x,
                                   p_y,
                                   s=300,
                                   marker='*',
                                   color='black',
                                   edgecolors='white',
                                   linewidth=1.,
                                   zorder=3.
                                   )
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
                ax.set_ylim(bottom=.35, top=-.45)
                #pyplot.ylabel('Spearman correlation', fontsize=20, fontweight='bold')
                pyplot.ylabel('Impact of predictor on predictions', fontsize=20, fontweight='bold')
                pyplot.yticks(fontsize=15)
                pyplot.savefig(os.path.join(curr_out_f, '{}_{}_{}_{}.jpg'.format(fam, case, targ, pov)), dpi=300)
                pyplot.clf()
                pyplot.close()
                with open(os.path.join(curr_out_f, '{}_{}_{}_{}_stats.tsv'.format(fam, case, targ, pov)), 'w') as o:
                    o.write('target_type\tpredictor\ttarget_time_point\tavg_correlation\tfdr_corrected_p_value\traw_p_value\n')
                    for x_name, corr_p, raw_p, y in zip(x_names, corr_ps, ps, ys):
                        o.write('{}\t{}\t{}\t'.format(fam, x_name, targ))
                        o.write('{}\t{}\t{}\n'.format(round(y, 5), round(corr_p, 5), round(raw_p, 5)))

