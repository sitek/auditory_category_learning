import os

POPULATE_INTENDED_FOR_OPTS = {
        'matching_parameters': ['ImagingVolume', 'Shims', 'ModalityAcquisitionLabel'],
        'criterion': 'Closest'
         } 

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
    mp2rage_inv1mag = create_key('sub-{subject}/anat/sub-{subject}_inv-1_part-mag_MP2RAGE')
    mp2rage_inv1ph  = create_key('sub-{subject}/anat/sub-{subject}_inv-1_part-phase_MP2RAGE')
    mp2rage_inv2mag = create_key('sub-{subject}/anat/sub-{subject}_inv-2_part-mag_MP2RAGE')
    mp2rage_inv2ph  = create_key('sub-{subject}/anat/sub-{subject}_inv-2_part-phase_MP2RAGE')
    mp2rage_t1map   = create_key('sub-{subject}/anat/sub-{subject}_T1map')
    mp2rage_uniT1   = create_key('sub-{subject}/anat/sub-{subject}_UNIT1')
    #mp2rage_uniT1den = create_key('sub-{subject}/anat/sub-{subject}_UNIT1DEN')
    #mp2rage_div     = create_key('sub-{subject}/anat/sub-{subject}_DIV')
    #mp2rage_head    = create_key('sub-{subject}/anat/sub-{subject}_HeadMask')
    t1w             = create_key('sub-{subject}/anat/sub-{subject}_T1w')

    # functional paths done in BIDS format
    task_tonecat = create_key('sub-{subject}/func/sub-{subject}_task-tonecat_run-{item:02d}_bold')
    task_tonecat_sbref = create_key('sub-{subject}/func/sub-{subject}_task-tonecat_run-{item:02d}_sbref')
    task_stgrid = create_key('sub-{subject}/func/sub-{subject}_task-stgrid_run-{item:02d}_bold')
    task_stgrid_sbref = create_key('sub-{subject}/func/sub-{subject}_task-stgrid_run-{item:02d}_sbref')

    # fieldmap paths done in BIDS format
    fieldmap_se_ap = create_key('sub-{subject}/fmap/sub-{subject}_acq-func_dir-AP_run-{item:02d}_epi')
    fieldmap_se_pa = create_key('sub-{subject}/fmap/sub-{subject}_acq-func_dir-PA_run-{item:02d}_epi')
    
    # SWI paths
    swi_mag = create_key('sub-{subject}/swi/sub-{subject}_part-mag_swi')
    swi_ph = create_key('sub-{subject}/swi/sub-{subject}_part-phase_swi')
    swi_mip = create_key('sub-{subject}/swi/sub-{subject}_rec-mIP_swi')
    swi = create_key('sub-{subject}/swi/sub-{subject}_swi')
    
    # create `info` dict
    info = {
            #mp2rage_inv1mag:[], mp2rage_inv1ph:[], 
            #mp2rage_inv2mag:[], mp2rage_inv2ph:[], 
            mp2rage_t1map:[], mp2rage_uniT1:[], 
            #mp2rage_uniT1den:[], 
            #mp2rage_div:[], mp2rage_head:[], 
            t1w:[],
            fieldmap_se_ap:[], fieldmap_se_pa:[],
            swi_mag:[], swi_ph:[], swi_mip:[], swi:[],
            task_tonecat:[], task_tonecat_sbref:[],
            task_stgrid:[], task_stgrid_sbref:[]}
    
    for s in seqinfo:
        # MP2RAGE anatomy run
        if ('MP2RAGE' in s.series_id):
            '''
            if ('INV1_PHS' in s.series_description):
                info[mp2rage_inv1ph] = [s.series_id]
            elif ('INV1' in s.series_description):
                info[mp2rage_inv1mag] = [s.series_id]
            elif ('INV2_PHS' in s.series_description):
                info[mp2rage_inv2ph] = [s.series_id]
            elif ('INV2' in s.series_description):
                info[mp2rage_inv2mag] = [s.series_id]
            '''
            if ('UNI-DEN' in s.series_description):
                #info[mp2rage_uniT1den] = [s.series_id]
                info[t1w] = [s.series_id]
            elif ('UNI_Images' in s.series_description):
                info[mp2rage_uniT1] = [s.series_id]
            
            #elif ('DIV' in s.series_description):
            #    info[mp2rage_div] = [s.series_id]
            #elif ('HeadMask' in s.series_description):
            #    info[mp2rage_head] = [s.series_id]
            
            elif ('T1_Images' in s.series_description):
                info[mp2rage_t1map] = [s.series_id]
        # functional runs
        for trx in range(1,7): # runs are numbered on the scanner
            # fMRI task: tone learning
            if ('Tone Learning %d'%trx in s.series_description):
                if ('FieldMap SE PA' in s.series_description):
                    if ~('SBRef' in s.series_description):
                        if ('M' in s.image_type):
                            info[fieldmap_se_pa].append(s.series_id)
                elif ('FieldMap SE AP' in s.series_description):
                    if ~('SBRef' in s.series_description):
                        if ('M' in s.image_type):
                            info[fieldmap_se_ap].append(s.series_id)
                elif ('SBRef' in s.series_description):
                    if ~('FieldMap SE' in s.series_description):
                        info[task_tonecat_sbref].append(s.series_id)
                elif (s.dim4 > 100): # a full run should have over 100 volumes
                        info[task_tonecat].append(s.series_id)

            # fMRI task: Spectrotemporal grid
            if ('STgrid %d'%trx in s.series_description):
                if ('FieldMap SE PA' in s.series_description):
                    if ~('SBRef' in s.series_description):
                        if ('M' in s.image_type):
                            info[fieldmap_se_pa].append(s.series_id)
                elif ('FieldMap SE AP' in s.series_description):
                    if ~('SBRef' in s.series_description):
                        if ('M' in s.image_type):
                            info[fieldmap_se_ap].append(s.series_id)
                elif ('SBRef' in s.series_description):
                    if ~('FieldMap SE' in s.series_description):
                        info[task_stgrid_sbref].append(s.series_id)
                elif (s.dim4 > 100): # a full run should have over 100 volumes
                        info[task_stgrid].append(s.series_id)

        # SWI (not yet in official BIDS standard)
        if ('SWI' in s.protocol_name):
            continue
            # commenting out for now due to heudiconv set issue
            # on multi-echo images - 
            # should be fixed by nipy/heudiconv #461
            #if ('Mag' in s.series_description):
            #    info[swi_mag].append(s.series_id)
            #if ('Pha' in s.series_description):
            #    info[swi_ph].append(s.series_id)
            #if ('mIP' in s.series_description):
            #    info[swi_mip].append(s.series_id)
            #if ('SWI' in s.series_description):
            #    info[swi].append(s.series_id)
            
    return info

