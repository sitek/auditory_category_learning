Processing and analyzing tone-learning fMRI data. WIP - KRS 2022.10

**Processing pipeline**
***Dicom conversion***
1. Peek at the dicom .tsv file  using `initialize_dicoms_heudiconv.sh`
2. Create `heuristic.py` based on your MRI sequences
3. Convert dicoms to .nii using `convert_dicoms_heudiconv.sh`

***MRI preprocessing***
1. Preprocess anatomical and functional MRI with `run_fmriprep.sh` 
(Note: this runs using a Singularity image, so may need to create that first)

***Behavioral data conversion ***
1. Run `convert_behav_to_bids.py` to get psychopy outputs into BIDS-compatible format

***Univariate analysis***
1. Run `univariate_analysis.py`

***Multivariate analysis***
1.Create trial-specific beta estimates with `modeling_firstlevel_singleevent.py` 
(Note: depending on the stimulus set, this will yield different results than `modeling_first_level_stimulus_perrun.py`. 
For our 16-stimulus set, we repeat each sound 3 times per run, so these outputs would be different. 
For the 40-stimulus set, each sound is only used once per run, so the estimates would be the same 
 [although the output names would be different].)
