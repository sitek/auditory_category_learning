#!/usr/bin/env python
# coding: utf-8

# Based on the rsatoolbox tutorial: https://rsatoolbox.readthedocs.io/en/stable/demo_searchlight.html
import os
import argparse
import sys

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import nibabel as nib
import seaborn as sns

from nilearn import plotting
from nilearn.image import new_img_like

from rsatoolbox.inference import eval_fixed
from rsatoolbox.model import ModelFixed, Model
from rsatoolbox.rdm import RDMs

from rsatoolbox.util.searchlight import get_volume_searchlight, get_searchlight_RDMs, evaluate_models_searchlight
from glob import glob

parser = argparse.ArgumentParser(
                description='Create subject-specific searchlight RSA',
                epilog=('Example: python rsa_searchlight.py --sub=FLT02 '
                        ' --space=MNI152NLin2009cAsym '
                        ' --analysis_window=run '
                        ' --fwhm=1.5 --searchrad=3'
                        ' --contrast=sound '
                        ' --mask_dir=/PATH/TO/MASK/DIR/ '                        
                        ' --bidsroot=/PATH/TO/BIDS/DIR/ ' 
                        ' --fmriprep_dir=/PATH/TO/FMRIPREP/DIR/'))

parser.add_argument("--sub", help="participant id", 
                    type=str)
parser.add_argument("--space", help="space label", 
                    type=str)
parser.add_argument("--analysis_window", 
                    help="analysis window (options: session, run}", 
                    type=str)
parser.add_argument("--fwhm", help="spatial smoothing full-width half-max", 
                    type=str)
parser.add_argument("--searchrad", help="radius of searchlight (in voxels)", 
                    type=int)
parser.add_argument("--contrast", help="contrast to analyze (options: sound, resp, fb)", 
                    type=str)
parser.add_argument("--mask_dir", 
                    help="directory containing subdirectories with masks for each subject", 
                    type=str)
parser.add_argument("--bidsroot", 
                    help="top-level directory of the BIDS dataset", 
                    type=str)
parser.add_argument("--fmriprep_dir", 
                    help="directory of the fMRIprep preprocessed dataset", 
                    type=str)
args = parser.parse_args()

if len(sys.argv) < 2:
    parser.print_help()
    print(' ')
    sys.exit(1)
    
sub_id          = args.sub
space_label     = args.space
analysis_window = args.analysis_window
fwhm           = args.fwhm
searchrad      = args.searchrad
contrast_label = args.contrast
mask_dir     = args.mask_dir
bidsroot     = args.bidsroot
fmriprep_dir = args.fmriprep_dir

# other directory definitions
deriv_dir = os.path.join(bidsroot, 'derivatives')
model_dir = os.path.join(deriv_dir, 'nilearn', 
                         'level-1_fwhm-{}'.format(fwhm))

print('participant ID: ', sub_id, 
      space_label, 
      '\nfirst-level FWHM: ', fwhm, 
      '\ndesired searchlight radius (in voxels): ', searchrad)

''' Helper functions '''
def upper_tri(RDM):
    """upper_tri returns the upper triangular index of an RDM

    Args:
        RDM 2Darray: squareform RDM

    Returns:
        1D array: upper triangular vector of the RDM
    """
    # returns the upper triangle
    m = RDM.shape[0]
    r, c = np.triu_indices(m, 1)
    return RDM[r, c]

def get_searchlight_rdm(mask_data, image_paths, centers, neighbors):
    # loop over all images
    x, y, z = mask_data.shape
    data = np.zeros((len(image_paths), x, y, z))
    for ix, im in enumerate(image_paths):
        #print(im)
        data[ix] = nib.load(im).get_fdata()

    # only one pattern per image
    image_value = np.arange(len(image_paths))

    # reshape data so we have n_observastions x n_voxels
    data_2d = data.reshape([data.shape[0], -1])
    data_2d = np.nan_to_num(data_2d)

    # Get RDMs – takes approx. 5 min
    # per https://github.com/rsagroup/rsatoolbox/issues/248#issuecomment-1437358066: 
    # only works with method='euclidean' if mask includes some 0s
    print('getting searchlight RDMs')
    SL_RDM = get_searchlight_RDMs(data_2d, centers, neighbors, 
                                  image_value, method='euclidean')
    
    return SL_RDM, data

def create_RDM_img(test_model, SL_RDM, data, mask_img):
    # takes a couple minutes to start running – don't give up too early!
    # in total, takes about 15 minutes to run with 2 cores
    print('Comparing searchlight RDMs')
    eval_results = evaluate_models_searchlight(SL_RDM, 
                                               test_model, 
                                               eval_fixed, 
                                               method='spearman', 
                                               n_jobs=-1)

    # get the evaulation score for each voxel
    # We only have one model, but evaluations returns a list. 
    # By using float we just grab the value within that list
    eval_score = [float(e.evaluations) for e in eval_results]

    # Create an 3D array, with the size of mask, and
    x, y, z = data.shape[1:]
    RDM_brain = np.zeros([x*y*z])
    RDM_brain[list(SL_RDM.rdm_descriptors['voxel_index'])] = eval_score
    RDM_brain = RDM_brain.reshape([x, y, z])

    plot_img = new_img_like(mask_img, RDM_brain)
    
    return plot_img

''' Make models '''
pattern_descriptors = {'tone': ['T1', 'T1', 'T1', 'T1', 
                                'T2', 'T2', 'T2', 'T2', 
                                'T3', 'T3', 'T3', 'T3', 
                                'T4', 'T4', 'T4', 'T4', ],
                       'talker': ['M1', 'M2', 'F1', 'F2',
                                  'M1', 'M2', 'F1', 'F2',
                                  'M1', 'M2', 'F1', 'F2',
                                  'M1', 'M2', 'F1', 'F2', ],
                      }
'''
# ### Stimulus RDMs
print('loading stimulus dissimilarity matrices')
stim_rdm_dir = os.path.join('/bgfs/bchandrasekaran/ngg12/',
                            '16tone/analysis_scripts/RDMs_kevin')

stim_rdms = sorted(glob(stim_rdm_dir+'/STIM*PCA*'))
n_rdms = len(stim_rdms)
stim_rdms_name_list = []
stim_rdms_array = np.zeros((n_rdms, 16, 16))
for i, fpath in enumerate(stim_rdms):
    rdm_name = os.path.basename(fpath)[5:-4]
    stim_rdm_data = np.genfromtxt(fpath, delimiter=',', skip_header=1)
    try:
        stim_rdms_array[i] = stim_rdm_data
        stim_rdms_name_list.append(rdm_name)
    except ValueError:
        # some of the DSMs are 4x4 instead of 16x16
        # so skip them
        continue

model_rdms = RDMs(stim_rdms_array,
                  rdm_descriptors={'stimulus_model':stim_rdms_name_list,},
                  pattern_descriptors=pattern_descriptors,
                  dissimilarity_measure='Euclidean'
                  )

# #### Convert to models
stim_models = []
for dx, descrip in enumerate(model_rdms.rdm_descriptors['stimulus_model']):
    spec_model = ModelFixed( '{} RDM'.format(descrip), 
                            model_rdms.subset('stimulus_model', descrip))
    stim_models.append(spec_model)
'''

# ### FFR RDMs
ffr_models = []
'''
ffr_strategy = 'group'
if ffr_strategy == 'group':
    # start with grand average FFR
    print('loading FFR dissimilarity matrix')
    ffr_rdm_fpath = os.path.join(stim_rdm_dir, 'FFRdistancesgrandavg.csv')
    rdm_name = 'FFR_grandavg'
    ffr_rdm_data = np.genfromtxt(ffr_rdm_fpath, delimiter=',', skip_header=1)
    print(len(ffr_rdm_data))
    
elif ffr_strategy == 'participant':
    # get participant-specific FFR distances
    participants_fpath = os.path.join(bidsroot, 'participants.tsv')
    participants_df = pd.read_csv(participants_fpath, sep='\t', dtype=str)

    # subjects to ignore (not fully processed, etc.)
    ignore_subs = []
    participants_df.drop(participants_df[participants_df.participant_id.isin(ignore_subs)].index,
                         inplace=True)

    # re-sort by participant ID
    participants_df.sort_values(by=['participant_id'], 
                                ignore_index=True, 
                                inplace=True)
    
    # get the FFR ID for the current participant ID
    ffr_id = participants_df.loc[participants_df['participant_id']=='sub-'+sub_id].FFR_id.item()
    
    ffr_rdm_fname = 'FFRdistances_{}_Man.csv'.format(ffr_id)
    ffr_rdm_fpath = os.path.join(stim_rdm_dir, ffr_rdm_fname)
    rdm_name = 'FFR_participant'
    ffr_rdm_data = np.genfromtxt(ffr_rdm_fpath, delimiter=',', skip_header=1)
    print(len(ffr_rdm_data))

# input array needs to be 3-dimensional, despite docs saying 2-D is ok
# (thus the newaxis)
ffr_rdm = RDMs(ffr_rdm_data[np.newaxis,:,:],
               rdm_descriptors={'FFR model': rdm_name},
               pattern_descriptors=pattern_descriptors,
               dissimilarity_measure='Euclidean')
ffr_model = ModelFixed('FFR_participant model', ffr_rdm)

ffr_models = [ffr_model]
'''
    
# ### Categorical RDMs

# make categorical RDMs
tone_rdm = np.array([[0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,],
                     [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,],
                     [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,],
                     [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,],
                     [1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1,],
                     [1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1,],
                     [1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1,],
                     [1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1,],
                     [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1,],
                     [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1,],
                     [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1,],
                     [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1,],
                     [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0,],
                     [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0,],
                     [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0,],
                     [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0,], ])

talker_rdm = np.array([[0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, ],
                       [1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, ],
                       [1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, ],
                       [1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, ],
                       [0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, ],
                       [1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, ],
                       [1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, ],
                       [1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, ],
                       [0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, ],
                       [1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, ],
                       [1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, ],
                       [1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, ],
                       [0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, ],
                       [1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, ],
                       [1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, ],
                       [1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, ], ])

rdms_array = np.array([tone_rdm, talker_rdm])

model_rdms = RDMs(rdms_array,
                  rdm_descriptors={'categorical_model':['tone', 'talker'],},
                  pattern_descriptors=pattern_descriptors,
                  dissimilarity_measure='Euclidean'
                 )

tone_rdms = model_rdms.subset('categorical_model','tone')
talker_rdms = model_rdms.subset('categorical_model','talker')

# #### Convert from RDM to Model
tone_model = ModelFixed( 'Tone RDM', model_rdms.subset('categorical_model', 
                                                       'tone'))
talker_model = ModelFixed( 'Talker RDM', model_rdms.subset('categorical_model', 
                                                           'talker'))
cat_models = [tone_model, talker_model]

# ## Merge model lists
#all_models = stim_models + ffr_models + cat_models
all_models = cat_models

''' Get searchlight and RDMs '''
mask_fpath = os.path.join(mask_dir, 
                          'sub-{}'.format(sub_id),
                          'space-{}'.format(space_label), 
                          'masks-dseg',
                          'sub-{}_space-{}_mask-gm.nii.gz'.format(sub_id, space_label))

mask_img = nib.load(mask_fpath)
mask_data = mask_img.get_fdata()
x, y, z = mask_data.shape

# takes about 10 minutes with 2,540,000 voxels; 
# grey matter-masked (449,000), about 3 min
print('getting searchlight voxels')
centers, neighbors = get_volume_searchlight(mask_data, 
                                            radius=searchrad, 
                                            threshold=0.5)


if analysis_window == 'session':
    model_desc = 'run-all_LSS'
    # set this path to wherever you saved the folder containing the img-files
    data_folder = os.path.join(model_dir, 
                               'sub-{}_space-{}'.format(sub_id, space_label),
                               model_desc)

    print(data_folder)
    image_paths = sorted(glob(f'{data_folder}/*contrast-{contrast_label}*map-tstat.nii.gz'))
    assert len(image_paths)

    SL_RDM, data = get_searchlight_rdm(mask_data, image_paths, centers, neighbors)

    # define output path
    out_dir = os.path.join(model_dir, 
                           'sub-{}_space-{}'.format(sub_id, space_label),
                           'rsa-searchlight_fwhm-{}_searchvox-{}_{}'.format(fwhm, searchrad, model_desc))
    if not os.path.exists(out_dir):
            os.makedirs(out_dir)

    # ## Compare RDMs
    for mi, test_model in enumerate(all_models): # cat_models

        plot_img = create_RDM_img(test_model, SL_RDM, data, mask_img)

        model_id = test_model.name.split(' ')[0]

        # #### Save correlation image
        sub_outname = 'sub-{}_fwhm-{}_searchvox-{}_rsa-searchlight_contrast-{}_model-{}.nii.gz'.format(sub_id, 
                                                                                            fwhm,
                                                                                            searchrad,
                                                                                            contrast_label,
                                                                                            model_id)
        out_fpath = os.path.join(out_dir, sub_outname)
        nib.save(plot_img, out_fpath)
        print('saved image to ', out_fpath)
        
elif analysis_window == 'run':
    model_desc = 'stimulus_per_run_LSS'
    # set this path to wherever you saved the folder containing the img-files
    sub_model_folder = os.path.join(model_dir, 
                               'sub-{}_space-{}'.format(sub_id, space_label),
                               model_desc)
    print(sub_model_folder)
    run_labels = [os.path.basename(x) for x in sorted(glob(sub_model_folder+'/run*'))]
    print(run_labels)
    
    for rx, run_label in enumerate(run_labels):
        # set this path to wherever you saved the folder containing the img-files
        data_folder = os.path.join(sub_model_folder, run_label)
        print('creating searchlight RDMs for ', run_label)
        image_paths = sorted(glob(f'{data_folder}/*contrast-{contrast_label}*map-tstat.nii.gz'))
        assert len(image_paths)
        print(image_paths)

        SL_RDM, data = get_searchlight_rdm(mask_data, image_paths, centers, neighbors)

        # define output path
        out_dir = os.path.join(model_dir, 
                               'sub-{}_space-{}'.format(sub_id, space_label),
                               'rsa-searchlight_fwhm-{}_searchvox-{}_{}'.format(fwhm, searchrad, model_desc),
                               run_label, )
        if not os.path.exists(out_dir):
                os.makedirs(out_dir)

        # ## Compare RDMs
        for mi, test_model in enumerate(all_models): # cat_models
            plot_img = create_RDM_img(test_model, SL_RDM, data, mask_img)

            model_id = test_model.name.split(' ')[0]

            # #### Save correlation image
            sub_outname = 'sub-{}_{}_fwhm-{}_searchvox-{}_rsa-searchlight_contrast-{}_model-{}.nii.gz'.format(sub_id, 
                                                                                                  run_label,
                                                                                                  fwhm,
                                                                                                  searchrad,
                                                                                                  contrast_label,
                                                                                                  model_id)
            out_fpath = os.path.join(out_dir, sub_outname)
            nib.save(plot_img, out_fpath)
            print('saved image to ', out_fpath)