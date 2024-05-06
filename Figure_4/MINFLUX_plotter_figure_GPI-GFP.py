import numpy as np
import matplotlib.pyplot as plt

import sys
sys.path.append(r'/Users/jakobrentsch/FU Box/Python/pythonProject1/MINFLUX/MINFLUX analysis') #directory of the MINFLUX analysis script
import MINFLUX_analysis_v3 as mfa

path = r'/Users/jakobrentsch/FU Box/Papers/MINFLUX/figure 4/python'
#path = r'C:\Users\Jakob\box.fu\Papers\MINFLUX\mat_files_jakob\python tests'
filename_msd = 'msd_all'
filename_D = 'diffusion_array'

msd_all = np.loadtxt(path + '/' + filename_msd + '.csv', delimiter=',', skiprows=1)
diffusion_array = np.loadtxt(path + '/' + filename_D + '.csv', delimiter=',', skiprows=1)
########################################################################################################################

[fig1, fig2, fig3, fig4, fig5, fig6, fig7, fig8, fig9] = mfa.hist_plotter(msd_all, diffusion_array, 500,
                                                                          'arial', 15, path)

plt.show(block=True)  # Set block=True

