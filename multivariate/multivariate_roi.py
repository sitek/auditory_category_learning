
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
                        ' --space=MNI152NLin2009cAsym --fwhm=1.5 --cond=tone '
                        ' --searchrad=9 --maptype=tstat '
                        ' --bidsroot=/PATH/TO/BIDS/DIR/ '
                        '--fmriprep_dir=/PATH/TO/FMRIPREP/DIR/')
        )

parser.add_argument("--sub", help="participant id", 
                    type=str)
parser.add_argument("--space", help="space label", 
                    type=str)
parser.add_argument("--analysis_window", help="analysis window (options: session, run}", 
                    type=str)
parser.add_argument("--fwhm", help="spatial smoothing full-width half-max", 
                    type=float)
parser.add_argument("--cond", help="condition to analyze", 
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
searchlight_radius = args.searchrad
maptype            = args.maptype
bidsroot           = args.bidsroot
fmriprep_dir       = args.fmriprep_dir

''' define other inputs '''
nilearn_dir = os.path.join(bidsroot, 'derivatives', 'nilearn')

roi_list = ['L-IC', 'L-MGN',
            'L-HG', 'L-PP', 'L-PT', 'L-STGa', 'L-STGp', 
            'R-IC', 'R-MGN',
            'R-HG', 'R-PP', 'R-PT', 'R-STGa', 'R-STGp', ]

model_desc = 'stimulus_per_run_LSS'

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

def sub_region_svc(sub_id, mask_fpath, stat_maps, cv, fwhm=1.5, space_label='T1w'):
    from sklearn.svm import SVC
    from sklearn.model_selection import cross_val_score
    from sklearn.model_selection import permutation_test_score

    #print(sub_id)
    #print(os.path.basename(mask_fpath))
          
    conditions_tone, conditions_talker, conditions_all, conditions_shuffled = create_labels(stat_maps)

    # extract beta values from stat maps for given region of interest
    fmri_masked, masker = mask_fmri(stat_maps, mask_fpath, fwhm_sub)

    svc = SVC()

    tone_cv_scores = cross_val_score(svc, fmri_masked, conditions_tone, cv=cv)
    print("Tone SVC accuracy: {:.3f} with a standard deviation of {:.2f}".format(tone_cv_scores.mean(),tone_cv_scores.std()))

    talker_cv_scores = cross_val_score(svc, fmri_masked, conditions_talker, cv=cv)
    print("Talker SVC accuracy: {:.3f} with a standard deviation of {:.2f}".format(talker_cv_scores.mean(),talker_cv_scores.std()))

    null_cv_scores = permutation_test_score(svc, fmri_masked, conditions_tone, cv=cv, )[1]
    print("Permutation test score: {:.3f} with a standard deviation of {:.2f}".format(null_cv_scores.mean(),null_cv_scores.std()))
    
    #plot_confusion_matrix(fmri_masked, conditions_tone, mask_descrip, sub_id)
    #plot_confusion_matrix(fmri_masked, conditions_talker, mask_descrip, sub_id)
    
    return fmri_masked, tone_cv_scores, talker_cv_scores, null_cv_scores

def plot_confusion_matrix(X, y, mask_descrip, sub_id):
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

    sub_title = 'sub-%s mask-%s'%(sub_id, mask_descrip)

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
print(sub_id)

fig, axs = plt.subplots(nrows=2, ncols=round(num_rois/2), figsize=(num_rois,6), dpi=300)
fig.suptitle(sub_id)

nilearn_sub_dir = os.path.join(bidsroot, 'derivatives', 'nilearn', 
                                   'level-1_fwhm-%.02f'%fwhm_sub, 
                                   'sub-%s_space-%s'%(sub_id, space_label))

# run-specific stimulus stat maps
stat_maps = sorted(glob(nilearn_sub_dir+f'/{model_desc}/run*/*di*map-{maptype}.nii.gz')) 
print('# of stat maps: ', len(stat_maps))    
#print(stat_maps)

# generate condition labels based on filenames
conditions_tone, conditions_talker, conditions_all, conditions_shuffled = create_labels(stat_maps)

for mx, mask_descrip in enumerate(roi_list):
    # define the mask for the region of interest
    print(mask_descrip)
    masks_dir = os.path.join(nilearn_dir, 'masks', 'sub-%s'%sub_id, 'space-%s'%space_label)
    mask_fpath = glob(masks_dir + '/masks-*/' + 'sub-%s_space-%s_mask-%s.nii.gz'%(sub_id, space_label, mask_descrip))[0]

    fmri_masked, masker = mask_fmri(stat_maps, mask_fpath, fwhm_sub)

    # Split the data into a training set and a test set
    x, y = fmri_masked, conditions_tone
    '''
    X_train, X_test, y_train, y_test = train_test_split(x,y, 
                                                        test_size=0.25,
                                                        random_state=0, 
                                                        stratify=y)
    '''
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
    np.savetxt('sub-{}_mask-{}_label-{}_confusion_matrix.csv'.format(sub_id, mask_descrip, 'tone'), cm)
    
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
fig.savefig(f'sub-{sub_id}_striatal_label-{cond_label}_confusion_matrices.png')