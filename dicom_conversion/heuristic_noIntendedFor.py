import os

def create_key(template, outtype=('nii.gz',), annotation_classes=None):
    if template is None or not template:
        raise ValueError('Template must be a valid format string')
    return template, outtype, annotation_classes


def infotodict(seqinfo):
    """Heuristic evaluator for determining which runs belong where

    allowed template fields - follow python string module:

    item: index within category
    subject: participant id
    seqitem: run number during scanning
    subindex: sub index within group
    """
    # mp2rage paths done in BIDS format
    mp2rage_t1map   = create_key('sub-{subject}/anat/sub-{subject}_T1map')
    mp2rage_uniT1   = create_key('sub-{subject}/anat/sub-{subject}_UNIT1')
    #mp2rage_uniT1den = create_key('sub-{subject}/anat/sub-{subject}_UNIT1DEN')
    t1w             = create_key('sub-{subject}/anat/sub-{subject}_T1w')

    # functional paths done in BIDS format
    task_tonecat = create_key('sub-{subject}/func/sub-{subject}_task-tonecat_run-{item:02d}_bold')
    task_tonecat_sbref = create_key('sub-{subject}/func/sub-{subject}_task-tonecat_run-{item:02d}_sbref')
    task_stgrid = create_key('sub-{subject}/func/sub-{subject}_task-stgrid_run-{item:02d}_bold')
    task_stgrid_sbref = create_key('sub-{subject}/func/sub-{subject}_task-stgrid_run-{item:02d}_sbref')

    # create `info` dict
    info = {mp2rage_t1map:[], mp2rage_uniT1:[], 
            t1w:[],
            task_tonecat:[], task_tonecat_sbref:[],
            task_stgrid:[], task_stgrid_sbref:[]}
    
    for s in seqinfo:
        # MP2RAGE anatomy run
        if ('MP2RAGE' in s.series_id):
            if ('UNI-DEN' in s.series_description):
                #info[mp2rage_uniT1den] = [s.series_id]
                info[t1w] = [s.series_id]
            elif ('UNI_Images' in s.series_description):
                info[mp2rage_uniT1] = [s.series_id]
            elif ('T1_Images' in s.series_description):
                info[mp2rage_t1map] = [s.series_id]
        # functional runs
        for trx in range(1,7): # runs are numbered on the scanner
            # fMRI task: tone learning
            if ('Tone Learning %d'%trx in s.series_description):
                if ('FieldMap' in s.series_description):
                    continue
                elif ('SBRef' in s.series_description):
                        info[task_tonecat_sbref].append(s.series_id)
                elif (s.dim4 > 100): # a full run should have over 100 volumes
                        info[task_tonecat].append(s.series_id)

            # fMRI task: Spectrotemporal grid
            if ('STgrid %d'%trx in s.series_description):
                if ('FieldMap' in s.series_description):
                    continue
                elif ('SBRef' in s.series_description):
                        info[task_stgrid_sbref].append(s.series_id)
                elif (s.dim4 > 100): # a full run should have over 100 volumes
                        info[task_stgrid].append(s.series_id)

    return info

