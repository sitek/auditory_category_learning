# WIP -- Pull data from the Pitt MRRC XNAT server
# using xnatpy -- NOT pyxnat! easily confusable
# KRS 2022.02.15

import xnat

# connect to xnat server
# also requires password (in email)
session=xnat.connect('https://xnat.mrrc.upmc.edu/', user='sitek', verify=False)

# check which experiments are available
list(session.projects['CHA-IBR'].experiments.values())

# pick the one you want by changing `exp`
exp = 0
experiment = list(session.projects['CHA-IBR'].experiments.values())[exp]

# check the acquired scans
experiment.scans

# download the data
experiment.subject.download_dir('/bgfs/bchandrasekaran/krs228/data/FLT/sourcedata/dicoms/')
