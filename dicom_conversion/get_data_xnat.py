# WIP -- Pull data from the Pitt MRRC XNAT server
# using xnatpy -- NOT pyxnat! easily confusable
# KRS 2022.02.15

import xnat

# connect to xnat server
# also requires password (in email)
session=xnat.connect('https://xnat.mrrc.upmc.edu/', user='sitek', verify=False)

# pick this project
project = session.projects['CHA-IBR']
dicom_dir = '/bgfs/bchandrasekaran/krs228/data/FLT/sourcedata/dicoms/'

# UPDATE WITH DESIRED SUBJECT
subject_id = 'FLT00'
project.subjects[subject_id]

project.subjects[subject_id].download_dir(dicom_dir)

''' ARCHIVE - older method '''
# check which experiments are available
list(project.experiments.values())

# pick the one you want by changing `exp`
exp = 0
experiment = list(session.projects['CHA-IBR'].experiments.values())[exp]

# check the acquired scans
experiment.scans

# download the data
experiment.subject.download_dir(dicom_dir)
