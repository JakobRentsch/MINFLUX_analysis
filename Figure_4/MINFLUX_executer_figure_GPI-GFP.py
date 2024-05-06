import numpy as np

import sys
sys.path.append(r'/Users/jakobrentsch/FU Box/Python/pythonProject1/MINFLUX/MINFLUX analysis') #directory of the MINFLUX analysis script
import MINFLUX_analysis_v3 as mfa

path = r'/Users/jakobrentsch/FU Box/Papers/MINFLUX/figure 4/python'
#path = r'C:\Users\Jakob\box.fu\Papers\MINFLUX\mat_files_jakob\python tests'
filename = 'clean'
array_raw = np.loadtxt(path + '/' + filename + '.csv', delimiter=',', skiprows=1)
########################################################################################################################

loc_array_tracklet = mfa.tracklet_splitter(array_raw, 10, path)
[msd_all, diffusion_array] = mfa.msd_calculater(loc_array_tracklet, 1, 5, 0, 50, path)

mfa.tracklet_diffusion(loc_array_tracklet, diffusion_array, path)

#print(np.loc_array[:, 5])

