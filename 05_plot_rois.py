import matplotlib
import nilearn
import numpy
import os

from matplotlib import pyplot
from matplotlib.colors import LinearSegmentedColormap
from nilearn import datasets, plotting, surface

#from utils.atlas_harvard_oxford_lobes import harvard_oxford_to_lobes

fsaverage = datasets.fetch_surf_fsaverage()
dataset = nilearn.datasets.fetch_atlas_harvard_oxford('cort-maxprob-thr25-2mm')
maps = dataset['maps']
maps_data = maps.get_fdata()
labels = dataset['labels']
collector = {l : numpy.array([], dtype=numpy.float64) for l in labels}

rel_names = {
             'lesions' : {
                    'ifg' : [
                            'Frontal Orbital Cortex',
                            ],
                    'ptl' : [
                            'Superior Temporal Gyrus, posterior division',
                            'Superior Temporal Gyrus, posterior division',
                            'Middle Temporal Gyrus, posterior division',
                              ],
                    'dlpfc' : [
                              'Middle Frontal Gyrus',
                              ]
                              },
             'activations' : {
                    'ifg' : [
                            'Frontal Orbital Cortex',
                            'Frontal_Inf_Orb_L',
                            'Frontal_Inf_Orb_R',
                            ],
                    'sma' : [
                            'Juxtapositional Lobule Cortex (formerly Supplementary Motor Cortex)',
                            'Supp_Motor_Area_L'
                            ],
                    'ptl' : [
                            'Superior Temporal Gyrus, posterior division',
                            'Superior Temporal Gyrus, posterior division',
                            'Middle Temporal Gyrus, posterior division',
                            'Temporal_Sup_L', 'Temporal_Sup_R',
                            'Temporal_Mid_L', 'Temporal_Mid_R',
                            'Temporal_Inf_L', 'Temporal_Inf_R',
                              ],
                    'dlpfc' : [
                              'Middle Frontal Gyrus',
                              'Frontal_Mid_L',
                              'Frontal_Mid_R'
                              ]
                              },
             }

for color in [
              'white',
              #'black',
              ]:
    colormaps = {
             'left_ifg' : LinearSegmentedColormap.from_list(
                                               "mycmap",
                                              [
                                               color,
                                               'paleturquoise',
                                                   ]),
             'right_ifg' : LinearSegmentedColormap.from_list(
                                               "mycmap",
                                              [
                                               color,
                                               'forestgreen'
                                                   ]),
             'left_ptl' : LinearSegmentedColormap.from_list(
                                               "mycmap",
                                              [
                                               color,
                                               'steelblue'
                                                   ]),
             'left_sma' : LinearSegmentedColormap.from_list(
                                               "mycmap",
                                              [
                                               color,
                                               'lightpink',
                                                   ]),
             'left_dlpfc' : LinearSegmentedColormap.from_list(
                                               "mycmap",
                                              [
                                               color,
                                               'mediumvioletred',
                                                   ]),
             'right_dlpfc' : LinearSegmentedColormap.from_list(
                                               "mycmap",
                                              [
                                               color,
                                               'darkseagreen'
                                                   ]),
             }

    for area, cmap in colormaps.items():
        label = area.split('_')[-1]
        relevant_labels = [i for i, l in enumerate(labels) if l in rel_names['activations'][label]]
        msk = numpy.array([1. if v in relevant_labels else 0. for v in maps_data.flatten()]).reshape(maps_data.shape)
        print(area)
        print(sum(msk.flatten()))
        atl_img = nilearn.image.new_img_like(maps, msk)
        out = os.path.join('rois_brains',)
        os.makedirs(out, exist_ok=True)
        ### Right
        if 'right' in area:
            texture = surface.vol_to_surf(atl_img, fsaverage.pial_right,
                                          interpolation='nearest',
                    )
            r = plotting.plot_surf_stat_map(
                        fsaverage.pial_right,
                        texture,
                        hemi='right',
                        title='{} - right hemisphere'.format(area),
                        threshold=0.05,
                        bg_map=None,
                        darkness=0.5,
                        cmap=cmap,
                        alpha=0.,
                        )
            r.savefig(os.path.join(out, \
                        '{}_{}.svg'.format(color, area),
                        ),
                        dpi=600
                        )
            pyplot.clf()
            pyplot.close()
        ### Left
        elif 'left' in area:
            texture = surface.vol_to_surf(atl_img,
                                          fsaverage.pial_left,
                                          interpolation='nearest',
                    )
            l = plotting.plot_surf_stat_map(
                        fsaverage.pial_left,
                        texture,
                        hemi='left',
                        title='{} - left hemisphere'.format(area),
                        colorbar=True,
                        threshold=0.05,
                        bg_map=None,
                        cmap=cmap,
                        darkness=0.5,
                        alpha=0.,
                        )
            l.savefig(os.path.join(out, \
                       '{}_{}.svg'.format(color, area),
                        ),
                        dpi=600
                        )
            pyplot.clf()
            pyplot.close()

area = 'bg'
cmap = LinearSegmentedColormap.from_list(
                                   "mycmap",
                                  [
                                   'white',
                                   'silver',
                                       ])
### Right
texture = surface.vol_to_surf(atl_img, fsaverage.pial_right,
                              interpolation='nearest',
        )
r = plotting.plot_surf_stat_map(
            fsaverage.pial_right,
            texture,
            hemi='right',
            title='{} - right hemisphere'.format(area),
            threshold=0.05,
            bg_map=None,
            darkness=0.5,
            cmap=cmap,
            alpha=0.1,
            )
r.savefig(os.path.join(out, \
            'right_{}.svg'.format(area),
            ),
            dpi=600
            )
pyplot.clf()
pyplot.close()
### Left
texture = surface.vol_to_surf(atl_img,
                              fsaverage.pial_left,
                              interpolation='nearest',
        )
l = plotting.plot_surf_stat_map(
            fsaverage.pial_left,
            texture,
            hemi='left',
            title='{} - left hemisphere'.format(area),
            colorbar=True,
            threshold=0.05,
            bg_map=None,
            cmap=cmap,
            darkness=0.5,
            alpha=0.1,
            )
l.savefig(os.path.join(out, \
            'left_{}.svg'.format(area),
            ),
            dpi=600
            )
pyplot.clf()
pyplot.close()
