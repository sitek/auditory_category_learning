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
                        ' --space=MNI152NLin2009cAsym --fwhm=1.5 '
                        ' --bidsroot=/PATH/TO/BIDS/DIR/ ' 
                        ' --fmriprep_dir=/PATH/TO/FMRIPREP/DIR/'))

parser.add_argument("--sub", help="participant id", 
                    type=str)
parser.add_argument("--space", help="space label", 
                    type=str)
parser.add_argument("--fwhm", help="spatial smoothing full-width half-max", 
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
    
sub_id = args.sub
space_label=args.space
fwhm = args.fwhm
bidsroot = args.bidsroot
fmriprep_dir = args.fmriprep_dir
print(sub_id, space_label, fwhm)

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

import matplotlib.colors
def RDMcolormapObject(direction=1):
    """
    Returns a matplotlib color map object for RSA and brain plotting
    """
    if direction == 0:
        cs = ['yellow', 'red', 'gray', 'turquoise', 'blue']
    elif direction == 1:
        cs = ['blue', 'turquoise', 'gray', 'red', 'yellow']
    else:
        raise ValueError('Direction needs to be 0 or 1')
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", cs)
    return cmap

deriv_dir = os.path.join(bidsroot, 'derivatives')

model_dir = os.path.join(deriv_dir, 'nilearn', 
                         'level-1_fwhm-{}'.format(fwhm))

# ## Make models
pattern_descriptors = {'tone': ['T1', 'T1', 'T1', 'T1', 
                                'T2', 'T2', 'T2', 'T2', 
                                'T3', 'T3', 'T3', 'T3', 
                                'T4', 'T4', 'T4', 'T4', ],
                       'talker': ['M1', 'M2', 'F1', 'F2',
                                  'M1', 'M2', 'F1', 'F2',
                                  'M1', 'M2', 'F1', 'F2',
                                  'M1', 'M2', 'F1', 'F2', ],
                      }

# ### Stimulus RDMs
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
    spec_model = ModelFixed( '{} RDM'.format(descrip), model_rdms.subset('stimulus_model', descrip))
    stim_models.append(spec_model)


# #### Categorical RDMs

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
tone_model = ModelFixed( 'Tone RDM', model_rdms.subset('categorical_model', 'tone'))
talker_model = ModelFixed( 'Talker RDM', model_rdms.subset('categorical_model', 'talker'))
cat_models = [tone_model, talker_model]

# ## Merge model lists
all_models = stim_models + cat_models

# ## Get searchlight and RDMs

# set this path to wherever you saved the folder containing the img-files
data_folder = os.path.join(model_dir, 
                           'sub-{}_space-{}'.format(sub_id, space_label),
                           'run-all')
print(data_folder)
image_paths = sorted(glob('{}/*contrast-sound*map-tstat.nii.gz'.format(data_folder)))
assert len(image_paths)

mask_fpath = os.path.join(deriv_dir, 'nilearn', 'masks', 'sub-{}'.format(sub_id),
                          'space-{}'.format(space_label), 'masks-dseg',
                          'sub-{}_space-{}_mask-gm.nii.gz'.format(sub_id, space_label))

mask_img = nib.load(mask_fpath)
mask_data = mask_img.get_fdata()
x, y, z = mask_data.shape

# loop over all images
data = np.zeros((len(image_paths), x, y, z))
for x, im in enumerate(image_paths):
    #print(im)
    data[x] = nib.load(im).get_fdata()

# only one pattern per image
image_value = np.arange(len(image_paths))

# takes about 10 minutes with 2,540,000 voxels; grey matter-masked (449,000), about 3 min
centers, neighbors = get_volume_searchlight(mask_data, radius=5, threshold=0.5)

# reshape data so we have n_observastions x n_voxels
data_2d = data.reshape([data.shape[0], -1])
data_2d = np.nan_to_num(data_2d)

# Get RDMs – takes approx. 5 min
# per https://github.com/rsagroup/rsatoolbox/issues/248#issuecomment-1437358066: 
# only works with method='euclidean' if mask includes some 0s
SL_RDM = get_searchlight_RDMs(data_2d, centers, neighbors, image_value, method='euclidean')


# ## Compare RDMs
for mi, cat_model in enumerate(cat_models):
    model_id = cat_model.name.split(' ')[0]

    # takes a couple minutes to start running – don't give up too early!
    # in total, takes about 15 minutes to run with 2 cores
    print('Comparing searchlight RDMs with {}'.format(model_id))
    eval_results = evaluate_models_searchlight(SL_RDM, cat_model, eval_fixed, 
                                               method='spearman', n_jobs=-1)

    # get the evaulation score for each voxel
    # We only have one model, but evaluations returns a list. 
    # By using float we just grab the value within that list
    eval_score = [float(e.evaluations) for e in eval_results]


    # Create an 3D array, with the size of mask, and
    x, y, z = data.shape[1:]
    RDM_brain = np.zeros([x*y*z])
    RDM_brain[list(SL_RDM.rdm_descriptors['voxel_index'])] = eval_score
    RDM_brain = RDM_brain.reshape([x, y, z])

    # lets plot the voxels above the 99th percentile
    p_thresh = 95
    threshold = np.percentile(eval_score, p_thresh)
    print('{}% threshold = {}'.format(p_thresh, threshold))
    plot_img = new_img_like(mask_img, RDM_brain)

    # #### Save correlation image
    out_dir = os.path.join(model_dir, 
                           'sub-{}_space-{}'.format(sub_id, space_label),
                           'rsa-searchlight')
    if not os.path.exists(out_dir):
            os.makedirs(out_dir)
    sub_outname = 'sub-{}_rsa-searchlight_model-{}.nii.gz'.format(sub_id, model_id)
    out_fpath = os.path.join(out_dir, sub_outname)
    nib.save(plot_img, out_fpath)
    print('saved image to ', out_fpath)
