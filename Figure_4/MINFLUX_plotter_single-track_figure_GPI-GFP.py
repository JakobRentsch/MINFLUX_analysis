import numpy as np
import matplotlib.pyplot as plt

import sys
sys.path.append(r'/Users/jakobrentsch/FU Box/Python/pythonProject1/MINFLUX/MINFLUX analysis') #directory of the MINFLUX analysis script
import MINFLUX_analysis_v3 as mfa

path = r'/Users/jakobrentsch/FU Box/Papers/MINFLUX/figure 4/python'
#path = r'C:\Users\Jakob\box.fu\Papers\MINFLUX\mat_files_jakob\python tests'

filename_locs = 'loc_array_tracklet'
filename_D = 'diffusion_array'

loc_array_tracklet = np.loadtxt(path + '/' + filename_locs + '.csv', delimiter=',', skiprows=1)
diffusion_array = np.loadtxt(path + '/' + filename_D + '.csv', delimiter=',', skiprows=1)
########################################################################################################################

fig = mfa.single_track_plotter(loc_array_tracklet, diffusion_array, 10, 1, 'alpha', 0, 2,
                         'cool', 'grey', 0.2, 'arial', 15, 0, 0, 0, path)

plt.show(block=True)  # Set block=True
print('Done')


