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

import rsatoolbox
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
                        ' --fwhm=1.5  --maptype=tstat '
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
parser.add_argument("--maptype", help="type of map to operate on (options: beta, tstat)", 
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
fwhm            = args.fwhm
maptype         = args.maptype
mask_dir        = args.mask_dir
bidsroot        = args.bidsroot
fmriprep_dir    = args.fmriprep_dir

# other directory definitions
deriv_dir = os.path.join(bidsroot, 'derivatives')
model_dir = os.path.join(deriv_dir, 'nilearn', 
                         'level-1_fwhm-{}'.format(fwhm))

print('participant ID: ', sub_id, 
      space_label, 
      '\nfirst-level FWHM: ', fwhm, )

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

'''
# ## Merge model lists
all_models = stim_models + ffr_models + cat_models
'''

## define ROIs
network_name = 'auditory' # 'tian_subcortical_S3' # auditory

if network_name == 'auditory':
    roi_list = [
                'L-IC', 'L-MGN', 'L-HG', 'L-PT',  'L-PP', 'L-STGp', 'L-STGa', 'L-ParsOp', 'L-ParsTri',
                'R-IC', 'R-MGN', 'R-HG', 'R-PT',  'R-PP', 'R-STGp', 'R-STGa', 'R-ParsOp', 'R-ParsTri', 
               ]
elif network_name == 'aud-striatal':
    roi_list = ['L-Caud', 'L-Put', 'L-IC', 'L-MGN',
                'L-HG', 'L-PP', 'L-PT', 'L-STGa', 'L-STGp', 
                'L-ParsOp', 'L-ParsTri',
                'R-Caud', 'R-Put','R-IC', 'R-MGN',
                'R-HG', 'R-PP', 'R-PT', 'R-STGa', 'R-STGp', 
                'R-ParsOp', 'R-ParsTri',
               ]
elif network_name == 'tian_subcortical_S3':
    roi_list = [
                'CAU-DA-lh', 'CAU-VA-lh', 'pCAU-lh', 
                'PUT-DA-lh', 'PUT-DP-lh', 'PUT-VA-lh', 'PUT-VP-lh',
                'aGP-lh', 'pGP-lh', 'NAc-core-lh', 'NAc-shell-lh',
                'CAU-DA-rh', 'CAU-VA-rh', 'pCAU-rh', 
                'PUT-DA-rh', 'PUT-DP-rh', 'PUT-VA-rh', 'PUT-VP-rh',
                'aGP-rh', 'pGP-rh', 'NAc-core-rh', 'NAc-shell-rh', 
               ]
    

''' Generate run-specific RDMs '''
if analysis_window == 'session':
    model_desc = 'run-all_LSS'

    # set this path to wherever you saved the folder containing the img-files
    data_folder = os.path.join(model_dir, 'masked_statmaps',
                               f'sub-{sub_id}',
                               'statmaps_masked',
                               model_desc)

    all_im_paths = sorted(glob(f'{data_folder}/*mask-*_cond-*_map-{maptype}.csv'))

    all_mask_data = []
    for mx, mask_descrip in enumerate(roi_list):
        print(mask_descrip)
        image_paths = sorted(glob(f'{data_folder}/*{mask_descrip}*/*_cond-*_map-{maptype}.csv'))

        try:
            n_vox = np.genfromtxt(image_paths[0]).shape[0]
        except IndexError:
            n_vox = 1

        data = np.zeros((len(image_paths), n_vox))
        for x, im in enumerate(image_paths):
            data[x] = np.genfromtxt(im)
        all_mask_data.append(data)

        roi_rdms = []
    for rx, mask_data in enumerate(all_mask_data):
        dataset = rsatoolbox.data.Dataset(mask_data, 
                                          descriptors={'participant': sub_id, 
                                                       'ROI': roi_list[rx], 
                                                       #'group': group_id
                                                      },)
        test_rdm = rsatoolbox.rdm.calc_rdm(dataset)
        roi_rdms.append(test_rdm)

    concat_rdms = rsatoolbox.rdm.rdms.concat(roi_rdms)

elif analysis_window == 'run':
    model_desc = 'stimulus_per_run_LSS'
    # set this path to wherever you saved the folder containing the img-files
    model_folder = os.path.join(model_dir, 'masked_statmaps',
                               f'sub-{sub_id}',
                               'statmaps_masked',
                                model_desc)
    run_roi_rdm_list = []
    for runx, data_folder in enumerate(sorted(glob('{}/run*'.format(model_folder)))):
        all_mask_data = []
        for mx, mask_descrip in enumerate(roi_list):
            print(mask_descrip)
            image_paths = sorted(glob(f'{data_folder}/*{mask_descrip}*/*_cond-*_map-{maptype}.csv'))
            
            try:
                n_vox = np.genfromtxt(image_paths[0]).shape[0]
            except IndexError:
                n_vox = 1

            data = np.zeros((len(image_paths), n_vox))
            for x, im in enumerate(image_paths):
                data[x] = np.genfromtxt(im)
            all_mask_data.append(data)

        # loop through mask data
        roi_rdms = []
        for rx, mask_data in enumerate(all_mask_data):
            dataset = rsatoolbox.data.Dataset(mask_data, 
                                              descriptors={'participant': sub_id, 
                                                           'run': os.path.basename(data_folder),
                                                           'ROI': roi_list[rx], 
                                                           #'group': group_id,
                                                          },)
            test_rdm = rsatoolbox.rdm.calc_rdm(dataset)
            #roi_rdms.append(test_rdm)
            run_roi_rdm_list.append(test_rdm)
        concat_rdms = rsatoolbox.rdm.rdms.concat(run_roi_rdm_list)
        #concat_rdms = rsatoolbox.rdm.rdms.concat(roi_rdms)
    
print(concat_rdms)
# rename pattern descriptors
concat_rdms.pattern_descriptors = pattern_descriptors

# save subject-level RDMs
out_dir = os.path.join(model_dir, 'rsa_roi', network_name)
os.makedirs(out_dir, exist_ok=True)
out_fpath = os.path.join(out_dir,
                         f'sub-{sub_id}_{network_name}_{model_desc}_rdms.hdf5')
concat_rdms.save(out_fpath, 
                 file_type='hdf5', overwrite=True)