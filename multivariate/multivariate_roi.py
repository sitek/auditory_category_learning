
import os
import sys
import json
import argparse
import numpy as np
import nibabel as nib

from glob import glob
from nilearn.image import new_img_like

from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(
                description='Subject-level multivariate ROI analysis',
                epilog=('Example: python multivariate_roi.py --sub=FLT02 '
                        ' --space=MNI152NLin2009cAsym --fwhm=0.00 --cond=tone '
                        ' --searchrad=9 --maptype=tstat '
                        ' --bidsroot=/PATH/TO/BIDS/DIR/ '
                        '--fmriprep_dir=/PATH/TO/FMRIPREP/DIR/'
                        ' --mask_dir=/PATH/TO/MASK/DIR/ '
                        )
        )

parser.add_argument("--sub", help="participant id", 
                    type=str)
parser.add_argument("--space", help="space label", 
                    type=str)
parser.add_argument("--analysis_window", help="analysis window (options: session, run}", 
                    type=str)
parser.add_argument("--fwhm", help="spatial smoothing full-width half-max", 
                    type=float)
parser.add_argument("--cond", help="condition to analyze (options: assigned, shuffled)", 
                    type=str)
parser.add_argument("--contrast", help="contrast to analyze (options: sound, resp, fb)", 
                    type=str)
parser.add_argument("--searchrad", help="searchlight radius (in voxels)", 
                    type=str)
parser.add_argument("--maptype", help="type of map to operate on (options: beta, tstat)", 
                    type=str)
parser.add_argument("--bidsroot", 
                    help="top-level directory of the BIDS dataset", 
                    type=str)
parser.add_argument("--fmriprep_dir", 
                    help="directory of the fMRIprep preprocessed dataset", 
                    type=str)
parser.add_argument("--mask_dir", 
                    help="directory containing subdirectories with masks for each subject", 
                    type=str)

args = parser.parse_args()

if len(sys.argv) < 2:
    parser.print_help()
    print(' ')
    sys.exit(1)
    
subject_id         = args.sub
space_label        = args.space
analysis_window    = args.analysis_window
fwhm               = args.fwhm
cond_label         = args.cond
contrast_label     = args.contrast
searchlight_radius = args.searchrad
maptype            = args.maptype
bidsroot           = args.bidsroot
fmriprep_dir       = args.fmriprep_dir
mask_dir           = args.mask_dir

''' define other inputs '''
nilearn_dir = os.path.join(bidsroot, 'derivatives', 'nilearn')

split_design = 'random' 
model_desc = 'stimulus_per_run_LSS' # 'stimulus_per_run_LSS', 'trial_models_LSS'


''' define regions of interest '''
network_name = 'tian_subcortical_S3' # auditory

if network_name == 'auditory':
    roi_list = [
                'L-IC', 'L-MGN', 'L-HG', 'L-PT',  'L-PP', 
                'L-STGp', 'L-STGa', 'L-ParsOp', 'L-ParsTri',
                'R-IC', 'R-MGN', 'R-HG', 'R-PT',  'R-PP', 
                'R-STGp', 'R-STGa', 'R-ParsOp', 'R-ParsTri', 
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
num_rois = len(roi_list)
    
# output directory
out_dir = os.path.join(
                       nilearn_dir, 
                       'level-1_fwhm-%.02f'%fwhm,
                       'level-1_den_mvpc-roi',
                       network_name,
                       f'contrast-{contrast_label}')
os.makedirs(out_dir, exist_ok=True)
   

''' helper functions '''
def create_labels(stat_maps):
    import os
    from glob import glob
    from numpy.random import shuffle
    from copy import copy
    
    # all-stimulus decoding
    conditions_all = ['_'.join(os.path.basename(x).split('_')[5:8]) for x in (stat_maps)]

    # 4-category decoding
    conditions_tone = [os.path.basename(x).split('_')[5] for x in (stat_maps)]
    print('tone conditions: ', np.unique(conditions_tone))

    conditions_talker = [os.path.basename(x).split('_')[6] for x in (stat_maps)]
    print('talker conditions: ', np.unique(conditions_talker))
    
    # shuffled conditions
    conditions_shuffled = copy(conditions_tone)
    shuffle(conditions_shuffled)
    print('shuffled conditions: ', np.unique(conditions_shuffled))

    return conditions_tone, conditions_talker, conditions_all, conditions_shuffled

def mask_fmri(fmri_niimgs, mask_filename, fwhm):
    from nilearn.maskers import NiftiMasker
    masker = NiftiMasker(mask_img=mask_filename, #runs=session_label,
                         smoothing_fwhm=fwhm, standardize=True,
                         memory="nilearn_cache", memory_level=1)
    fmri_masked = masker.fit_transform(fmri_niimgs)
    return fmri_masked, masker

def plot_confusion_matrix(X, y, mask_descrip, subject_id):
    import numpy as np
    import matplotlib.pyplot as plt

    from sklearn import svm, datasets
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import ConfusionMatrixDisplay

    # Split the data into a training set and a test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25,
                                                        random_state=0, stratify=y)

    # Run classifier, using a model that is too regularized (C too low) to see
    # the impact on the results
    classifier = svm.SVC(kernel="linear", ).fit(X_train, y_train)

    np.set_printoptions(precision=2)

    sub_title = 'sub-%s mask-%s'%(subject_id, mask_descrip)

    # Plot non-normalized confusion matrix
    titles_options = [("Confusion matrix, without normalization", None),
                      ("Normalized confusion matrix", "true"),]
    #for title, normalize in titles_options:
    disp = ConfusionMatrixDisplay.from_estimator(
            classifier,
            X_test,
            y_test,
            #display_labels=class_names,
            cmap=plt.cm.Blues,
            #normalize='true', 
            )
    disp.ax_.set_title(sub_title)

    #print(title)
    #print(disp.confusion_matrix)

    #plt.show()
    
    return disp.confusion_matrix


''' run pipeline '''
print(subject_id)

fig, axs = plt.subplots(nrows=2, ncols=round(num_rois/2), figsize=(num_rois,6), dpi=300)
fig.suptitle(subject_id)

nilearn_sub_dir = os.path.join(nilearn_dir, 
                                   'level-1_fwhm-%.02f'%fwhm, 
                                   'sub-%s_space-%s'%(subject_id, space_label))

# run-specific stimulus stat maps
stat_maps = sorted(glob(nilearn_sub_dir+f'/{model_desc}/run*/*{contrast_label}*map-{maptype}.nii.gz')) 
print('# of stat maps: ', len(stat_maps))    
#print(stat_maps)

# generate condition labels based on filenames
conditions_tone, conditions_talker, conditions_all, conditions_shuffled = create_labels(stat_maps)
print(np.unique(conditions_tone))

for mx, mask_descrip in enumerate(roi_list):
    # define the mask for the region of interest
    print(mask_descrip)
    masks_dir = os.path.join(mask_dir, 'sub-%s'%subject_id, 'space-%s'%space_label)
    mask_fpath = glob(masks_dir + '/masks-*/' + f'sub-{subject_id}_space-{space_label}_mask-{mask_descrip}.nii.gz')[0]

    fmri_masked, masker = mask_fmri(stat_maps, mask_fpath, fwhm)

    # Split the data into a training set and a test set
    x, y = fmri_masked, conditions_tone
    if split_design == 'random':

        X_train, X_test, y_train, y_test = train_test_split(x,y, 
                                                            test_size=0.25,
                                                            random_state=0, 
                                                            stratify=y)

    elif split_design == 'talker-sex':
        # split into male and female talker stimuli
        stim_m_indices = [i for i, s in enumerate(conditions_all) if any(xs in s for xs in ['aN', 'bN'])]
        stim_f_indices = [i for i, s in enumerate(conditions_all) if any(xs in s for xs in ['hN', 'iN'])]

        # train male, test female
        X_train = x[stim_m_indices]
        y_train = [y[i] for i in stim_m_indices]
        X_test = x[stim_f_indices]
        y_test = [y[i] for i in stim_f_indices]
        
    # Run classifier
    clf = svm.SVC(decision_function_shape='ovo').fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    clf_acc = accuracy_score(y_test, y_pred)
    print(f'accuracy: {clf_acc:.04f}')
    
    # save confusion matrix
    cm = confusion_matrix(y_test, y_pred, normalize='true',)
    cm_out_fpath = os.path.join(out_dir, 
                                f'sub-{subject_id}_mask-{mask_descrip}_contrast-{contrast_label}_label-{cond_label}_confusion_matrix.csv')
    np.savetxt(cm_out_fpath, cm)
    
    '''
    # add ROI confusion matrix to plot
    ax = plt.subplot(2, num_rois/2, mx + 1)
    cm_display = ConfusionMatrixDisplay.from_predictions(y_test, y_pred,
                                                         ax=ax, 
                                                          colorbar=False, 
                                                          im_kw={'cmap':'Blues'},
                                                          display_labels=['T1', 'T2', 'T3', 'T4'],
                                                          include_values=False)
    ax.set_title(mask_descrip)

# better spacing
fig.tight_layout()
fig.savefig(os.path.join(out_dir, f'sub-{subject_id}_network-{network_name}_contrast-{contrast_label}_label-{cond_label}_confusion_matrices.png'))
'''