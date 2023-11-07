Processing and analyzing tone-learning fMRI data. WIP - KRS 2022.10

## Processing pipeline

**Dicom conversion: `./dicom_conversion/`**
1. Peek at the dicom .tsv file  using `initialize_dicoms_heudiconv.sh`
2. Create `heuristic.py` based on your MRI sequences
3. Convert dicoms to .nii using `convert_dicoms_heudiconv.sh`

**MRI preprocessing: `./fmriprep/`**
1. Preprocess anatomical and functional MRI with `run_fmriprep.sh` 
> (Note: this runs using a Singularity image, so may need to create that first)

**Behavioral data conversion: `./behav_conversion/`**
1. Run `convert_behav_to_bids.py` to get psychopy outputs into BIDS-compatible format

**Univariate analysis: `./univariate/`**
1. Run `univariate_analysis.py`
2. Run `group_level.ipynb` for group-level GLM and output maps/figures

**Multivariate analysis: `./multivariate/`**
1. Create trial-specific beta estimates with `modeling_firstlevel_singleevent_LSS.py` 

> (Note: depending on the stimulus set, this will yield different results than `modeling_first_level_stimulus_perrun_LSS.py`. 
> For our 16-stimulus set, we repeat each sound 3 times per run, so these outputs would be different. 
> For the 40-stimulus set, each sound is only used once per run, so the estimates would be the same 
> [although the output names would be different].)

2. Create grey matter mask for searchlight using `make_gm_mask.py` (WIP)
3. Run whole-brain searchlight with `multivariate_searchlight.py`
4. Run region-based decoding with `confusion_matrix_plots.py`
5. (Work-in-progress) Group-level searchlight decoder statistics with `group_level_searchlight_WIP.ipynb`

**Representational similarity analysis: `./rsa/`**
1. Run whole-brain RSA searchlight
2. Run region-based RSA