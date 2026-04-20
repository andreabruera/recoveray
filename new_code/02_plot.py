import numpy
import os
import pickle
import random
import scipy
import matplotlib

from matplotlib import font_manager, pyplot

# Using Helvetica as a font
font_folder = os.path.join('..', '..', 'fonts')
if os.path.exists(font_folder):
    font_dirs = [font_folder, ]
    font_files = font_manager.findSystemFonts(fontpaths=font_dirs)
    for p in font_files:
        font_manager.fontManager.addfont(p)
    matplotlib.rcParams['font.family'] = 'Helvetica LT Std'

n_subjects = 47
iterations = 1001
folds = 50
proportion = 0.2
test_size = int(47*proportion)

### main plots
with open(os.path.join('pkls', 'full_results.pkl'), 'rb') as o:
    full_results = pickle.load(o)

out = os.path.join('plots', 'main_text')
os.makedirs(out, exist_ok=True)

case_targets = {
           'ability':  ['acute', 'subacute', 'early-chronic'],
           'improvement':  ['acute2subacute', 'acute2early-chronic', 'subacute2early-chronic'],
          }

for analysis_type in [
                      'functional',
                      'traditional',
                      ]:
    results = full_results[analysis_type]['results']
    ps = full_results[analysis_type]['ps']
    modalities = full_results[analysis_type]['details']['modalities']
    confounds = full_results[analysis_type]['details']['confounds']
    povs = full_results[analysis_type]['details']['povs']
    ### full model
    ### results for targets (always 3)
    colors = {
              ### functional
              'acute' : [255, 95, 0],
              'subacute' : [114, 93, 239],
              'early-chronic' : [221, 33, 125],
              ### traditional
              'age' : [0, 0, 0],
             'lesion' : [221, 221, 221],
              'acute_score' : [0, 114, 178],
              'subacute_score' : [86, 180, 233],
              'early-chronic_score' : [0, 158, 115],
             }

    if analysis_type == 'functional':
        povs_mapper = {'ability' :
                       {
                       2 : [-.3, 0, .3],
                       1 : [-.15, .15,],
                       0 : [0.],
                       },
                     'improvement' :
                       {
                       0 : [-.15, .15,],
                       1 : [-.3, 0, .3],
                       2 : [-.3, 0, .3],
                       },
                      }
    elif analysis_type == 'traditional':
        povs_mapper = {'ability' :
                       {
                       2 : [-.3, -.1, .1, .3],
                       1 : [-.2, 0, .2],
                       0 : [-.1, .1],
                       },
                     'improvement' :
                       {
                       0 : [-.15, .15],
                       1 : [-.2, 0., .2, .2],
                       2 : [-.2, 0, .2],
                       },
                      }
    for case, case_results in results.items():
        fig, ax = pyplot.subplots(
                                  figsize=(15, 10),
                                  tight_layout=True,
                                 )
        labels = list()
        for mode in modalities:
            ### raw plotting

            start = 0
            for targ_idx, targ in enumerate(case_targets[case]):
                if analysis_type == 'traditional':
                    ncols=2
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
                else:
                    ncols = 3
                for pov_idx, pov in enumerate(povs):
                    ### color and label
                    if analysis_type == 'functional':
                        chosen_color = colors[pov]
                        label='{} {}'.format(pov, mode).capitalize()
                    else:
                        chosen_color = colors[mode]
                        label=mode.replace('_', ' ').capitalize()
                    ### x-axis corrections, color corrections for connectivity
                    if mode == 'connectivity':
                        side='high'
                        mode_corr = 0.05
                        gen_corr = 0.05
                        chosen_color = tuple([min(255, rgb+80)/255 for rgb in chosen_color])
                    else:
                        side = 'low'
                        mode_corr = -0.05
                        gen_corr = -0.05
                        chosen_color = tuple([rgb/255 for rgb in chosen_color])

                    ### not interested in predicting the past from the future...
                    if case == 'ability':
                        if pov_idx > targ_idx:
                            continue
                    elif case == 'improvement':
                        ### slightly more complicated
                        check_idx = case_targets['ability'].index(targ.split('2')[1])
                        if pov_idx > check_idx:
                            continue
                    if analysis_type == 'traditional':
                        x_mapper_idx = modalities.index(mode)
                    else:
                        x_mapper_idx = pov_idx
                    '''
                    parts = ax.violinplot(
                                  #case_results[mode][targ_idx, pov_idx, :],
                                  case_results[mode][targ_idx, pov_idx, 1:],
                                  positions=[targ_idx+gen_corr+povs_mapper[case][targ_idx][x_mapper_idx]],
                                  #showmeans=True,
                                  showextrema=False,
                                  side=side,
                                  widths=0.2,
                                  )
                    for pc in parts['bodies']:
                        pc.set_edgecolor(chosen_color)
                        pc.set_facecolor(chosen_color)
                        pc.set_alpha(.6)
                    '''
                    real_avg=numpy.nanmean(case_results[mode][targ_idx, pov_idx, 0, :])
                    #real_std=numpy.nanstd(case_results[mode][targ_idx, pov_idx, 0, :])
                    real_std=scipy.stats.sem(case_results[mode][targ_idx, pov_idx, 0, :])
                    perm_avgs=numpy.nanmean(case_results[mode][targ_idx, pov_idx, 1:, :], axis=1)
                    assert perm_avgs.shape == (iterations-1, )
                    set_label = False
                    if label not in labels:
                        set_label = True
                    if set_label:
                        ax.scatter(
                                  x=targ_idx+mode_corr+povs_mapper[case][targ_idx][x_mapper_idx],
                                  y = real_avg,
                                  color = chosen_color,
                                  edgecolor='white',
                                  linewidth=1.5,
                                  zorder=3,
                                  s=300,
                                  marker='D',
                                  label=label,
                                  )
                        labels.append(label)
                    else:
                        ax.scatter(
                                  x=targ_idx+mode_corr+povs_mapper[case][targ_idx][x_mapper_idx],
                                  y = real_avg,
                                  color = chosen_color,
                                  edgecolor='white',
                                  linewidth=1.5,
                                  zorder=3,
                                  s=300,
                                  marker='D',
                                  )
                    ax.bar(
                          x=targ_idx+mode_corr+povs_mapper[case][targ_idx][x_mapper_idx],
                          height=real_avg,
                          color = chosen_color,
                          edgecolor='white',
                          linewidth=1.5,
                          zorder=2.5,
                          width=0.1,
                          alpha=0.6,
                          )
                    ax.errorbar(
                          x=targ_idx+mode_corr+povs_mapper[case][targ_idx][x_mapper_idx],
                          y=real_avg,
                          yerr=real_std,
                          capsize=4.,
                          ecolor ='black',
                          zorder=2.75,
                          )
                    ax.scatter(
                             x=[targ_idx+mode_corr+gen_corr+povs_mapper[case][targ_idx][x_mapper_idx]+(0.00125*random.randrange(-10, 10)) for _ in range(iterations-1)],
                              y=perm_avgs,
                              alpha=0.01,
                              s=100,
                              color=chosen_color,
                              linewidth=1,
                              edgecolor='white',
                              zorder=3,
                              )
                    #xs.append('target: {}\npov: {}\n'.format(targ, pov))
                    p = ps[case][mode][targ_idx, pov_idx, 1]
                    if p<0.005:
                        #print(p)
                        ax.scatter(
                                   x=targ_idx+mode_corr+povs_mapper[case][targ_idx][x_mapper_idx]-0.015,
                                   y=-.05,
                                   marker='*',
                                   color='black',
                                   s=100,
                                   zorder=3,
                                   )
                        ax.scatter(
                                   x=targ_idx+mode_corr+povs_mapper[case][targ_idx][x_mapper_idx]+0.015,
                                   y=-.05,
                                   marker='*',
                                   color='black',
                                   s=100,
                                   zorder=3,
                                   )
                    elif p<0.05:
                        ax.scatter(
                                   x=targ_idx+mode_corr+povs_mapper[case][targ_idx][x_mapper_idx],
                                   y=-.05,
                                   marker='*',
                                   color='black',
                                   s=100,
                                   zorder=3,
                                   )
                    start += 1
        ax.legend(fontsize=22.5, ncols=ncols,
                  #loc=1,
                  loc=(0.025, 1.),
                  borderpad=0.1,
                  labelspacing=0.5,
                  handletextpad=0.4,
                  borderaxespad=0.25,
                  columnspacing=1.,
                  )
        ax.set_ylim(bottom=-.2, top=.52)
        ax.spines[['right', 'bottom', 'top']].set_visible(False)
        #pyplot.xticks(ticks=range(start), labels=xs)
        xticks_mapper = {
                       'acute' : 'Acute',
                       'subacute' : 'Subacute',
                       'early-chronic' : 'Early chronic',
                       'acute2subacute' : 'Early',
                       'acute2early-chronic' : 'Long-term',
                       'subacute2early-chronic' : 'Late',
                       }
        pyplot.xticks(
                      ticks=range(3),
                      labels=[xticks_mapper[c] for c in case_targets[case]],
                      fontsize=30,
                      fontweight='bold',
                     )
        pyplot.yticks(fontsize=20)
        ax.text(
                x=1,
                y=.45,
                s='Language {}'.format(case),
                fontsize=35,
                fontweight='bold',
                ha='center', va='center',
                )
        end = 2.4
        ax.hlines(
                  y=0,
                  xmin=-.2,
                  xmax=end,
                  color='black',
                  zorder=1.
                  )
        ax.hlines(
                  y=[y*0.1 for y in range(-1, 6) if y!=0],
                  xmin=-.2,
                  xmax=end,
                  color='silver',
                  linestyle='dashed',
                  zorder=1,
                  alpha=0.6
                  )
        pyplot.ylabel('Spearman correlation', fontsize=25)
        pyplot.savefig(
                       os.path.join(out, '{}_{}_raw.jpg'.format(analysis_type, case)),
                       pad_inches=0.,
                       dpi=300,
                       )
        pyplot.clf()
        pyplot.close()
