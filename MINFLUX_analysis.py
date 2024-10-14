import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cm


test

'''
Tracklet splitter function:
    -splits tracks into tracklets depending on consistency of time elapsed between localisations (ddtime_cutoff)
    -inputs are MINFLUX data array in the format: [track id, time in s, efo in hz, x in nm, y in nm, z in nm], 
    ddtime cutoff in % (ddtime/dtime * 100), if path is specified output is saved there as .csv
    -output is MINFLUX data array appended with column containing all tracklets
'''
def tracklet_splitter(loc_array, ddtime_cutoff, path=''):

    track_list = np.unique(loc_array[:, 0])
    len_loc_array = loc_array[:, 0].shape[0]
    zeros_array = np.zeros((len_loc_array, 1), dtype=np.float64)
    loc_array_tracklet = np.concatenate((loc_array, zeros_array), axis=1)

    # convert locs to um
    loc_array_tracklet[:, 3] = loc_array_tracklet[:, 3] / 1000
    loc_array_tracklet[:, 4] = loc_array_tracklet[:, 4] / 1000
    loc_array_tracklet[:, 5] = loc_array_tracklet[:, 5] / 1000

    # make all locs start at 0 and be positiv
    loc_array_tracklet[:, 3] = loc_array_tracklet[:, 3] - np.min(loc_array_tracklet[:, 3])
    loc_array_tracklet[:, 4] = loc_array_tracklet[:, 4] - np.min(loc_array_tracklet[:, 4])
    loc_array_tracklet[:, 5] = loc_array_tracklet[:, 5] - np.min(loc_array_tracklet[:, 5])

    temp_tracklet_id = 0

    for n in range(track_list.shape[0]):

        temp_track_id = track_list[n]
        temp_track_array = loc_array_tracklet[loc_array_tracklet[:, 0] == temp_track_id, :]
        temp_track_t = temp_track_array[:, 1]
        temp_track_t_diff = np.diff(temp_track_t)
        temp_track_t_diff_diff = np.diff(temp_track_t_diff)
        temp_tracklet_id_array = np.zeros(temp_track_array.shape[0], dtype=np.float64)

        if temp_tracklet_id_array.shape[0] > 2:

            for m in range(temp_track_t_diff_diff.shape[0]):

                if temp_track_t_diff_diff[m]/temp_track_t_diff[m]*100 > ddtime_cutoff:

                    temp_tracklet_id += 1
                    temp_tracklet_id_array[m] = temp_tracklet_id

                else:

                    temp_tracklet_id_array[m] = temp_tracklet_id

                temp_tracklet_id_array[-1] = temp_tracklet_id
                temp_tracklet_id_array[-2] = temp_tracklet_id

        else:

            temp_tracklet_id_array = np.full(temp_tracklet_id_array.shape, temp_tracklet_id)

        temp_tracklet_id += 1

        loc_array_tracklet[loc_array_tracklet[:, 0] == temp_track_id, 6] = temp_tracklet_id_array


    if path:

        header_names = ['track id', 'time in s', 'efo in hz', 'x in um', 'y in um', 'z in um', 'tracklet id']
        loc_array_tracklet_with_header = np.vstack([header_names, loc_array_tracklet])

        np.savetxt(path + '/loc_array_tracklet' + '.csv',
                   loc_array_tracklet_with_header, delimiter=',', fmt="%s")

    return loc_array_tracklet



'''
MSD calculater function:
    -calculates length, MSD, alpha, D for all tracklets
    -inputs are MINFLUX data array with tracklet column (output of tracklet splitter function), lag, 
    max number of steps to be calculated (max_steps), minimum track length in locs (min_len), cutoff for tracks that have too much elapsed time between locs ins s
    (dtime_cutoff), if path is specified output is saved there as .csv
    -for calculation of D linear fit of first 3 lags in performed
    -tracklets that are shorter than max_steps + 1 are filtered out
    -outputs are 2 arrays:
        -msd array containing all msds for all lags for all tracklets in the format: tracklet_id, lag, msd, 
        std(msd) in um^2/s , delta_t in s, normalized delta_t in s (to lag)
        -diffusion array in the format: tracklet_id, alpha, d in um^2/s, log10(d), time_error in %
'''
def msd_calculater(loc_array_tracklet, lag, max_steps, min_len, dtime_cutoff, path=''):

    tracklet_list = np.unique(loc_array_tracklet[:, 6])
    num_tracklets = tracklet_list.shape[0]

    filter_counter = 0

    msd_all = np.full([max_steps * num_tracklets, 6], np.nan)
    diffusion_array = np.full([max_steps * num_tracklets, 7], np.nan)

    for n in range(num_tracklets):

        temp_tracklet_id = tracklet_list[n]
        temp_tracklet_array = loc_array_tracklet[loc_array_tracklet[:, 6] == temp_tracklet_id, :]

        #check if the min_len or the max_steps cutoff would lead to a higher cutoff and then use the higher cutoff
        if max_steps*lag > min_len:

            len_cutoff = max_steps*lag

        else:

            len_cutoff = min_len

        if temp_tracklet_array.shape[0] > len_cutoff:

            msd_tracklet = np.zeros([max_steps, 6], dtype=np.float64)

            for m in range(max_steps):
                dt = (m + 1) * lag

                delta_coords = temp_tracklet_array[dt:, 3:5] - temp_tracklet_array[:-dt, 3:5]
                delta_time = temp_tracklet_array[dt:, 1] - temp_tracklet_array[:-dt, 1]
                squared_displacement = np.sum(delta_coords ** 2, axis=1)  # dx^2+dy^2+dz^2

                msd_tracklet[m, 0] = temp_tracklet_id  # trackid
                msd_tracklet[m, 1] = dt  # lag
                msd_tracklet[m, 2] = np.mean(squared_displacement)  # MSD
                msd_tracklet[m, 3] = np.std(squared_displacement)  # std
                msd_tracklet[m, 4] = np.mean(delta_time)  # time diff in s
                msd_tracklet[m, 5] = msd_tracklet[m, 4] / dt  # normalized time diff in s

            if np.mean(msd_tracklet[:, 5]) < dtime_cutoff:
                msd_all[filter_counter * max_steps:filter_counter * max_steps + max_steps, :] = msd_tracklet

                time = np.mean(msd_tracklet[:, 5])
                error = np.std(msd_tracklet[:, 5]) / np.mean(msd_tracklet[:, 5]) * 100

                msd_for_fit = msd_tracklet[0:2, 2]
                lag_for_fit = msd_tracklet[0:2, 4]

                fit = np.polyfit(np.log10(lag_for_fit), np.log10(msd_for_fit), 1)

                alpha = fit[0]
                d = np.power(10, fit[1]) / 6
                logd = np.log10(d)

                diffusion_array[filter_counter, 0] = temp_tracklet_id
                diffusion_array[filter_counter, 1] = temp_tracklet_array.shape[0]
                diffusion_array[filter_counter, 2] = alpha
                diffusion_array[filter_counter, 3] = d
                diffusion_array[filter_counter, 4] = logd
                diffusion_array[filter_counter, 5] = time
                diffusion_array[filter_counter, 6] = error

            filter_counter += 1
    msd_all = msd_all[~np.isnan(msd_all[:, 0]), :]
    diffusion_array = diffusion_array[~np.isnan(diffusion_array[:, 0]), :]

    if path:

        header_names_msd_all = ['tracklet id', 'lag', 'MSD in um^2', 'Std(MSD) in um^2', 'dt in s', 'normalized dt in s']
        msd_all_with_header = np.vstack([header_names_msd_all, msd_all])

        np.savetxt(path + '/msd_all' + '.csv',
                   msd_all_with_header, delimiter=',', fmt="%s")

        header_names_diffusion_array = ['tracklet id', 'length in locs', 'alpha', 'D in um^2/s', 'log10(D)', 'time in s',
                                         'time error in %']
        diffusion_array_with_header = np.vstack([header_names_diffusion_array, diffusion_array])

        np.savetxt(path + '/diffusion_array' + '.csv',
                   diffusion_array_with_header, delimiter=',', fmt="%s")

    return msd_all, diffusion_array



'''
Tracklet diffusion function
    -adds values for alpha and D for all tracklets to columns
    -inputs are loc table with tracklets (output of tracklet_splitter), array with the diffusion parameters of
    tracklets (2nd outout of MSD calculator function), if path is specified output is saved there as .csv
    -output is array of locs with tracklet and 2 more columns (alpha, D), empty fields are nan
'''
def tracklet_diffusion(loc_array_tracklet, diffusion_array, path=""):

    zeros_array = np.full((loc_array_tracklet.shape[0], 2), np.nan)
    loc_array_tracklet_colored = np.concatenate((loc_array_tracklet, zeros_array), axis=1)

    for n in range(diffusion_array.shape[0]):
        temp_tracklet_id = diffusion_array[n, 0]
        temp_tracklet_alpha = diffusion_array[n, 2]
        temp_tracklet_D = diffusion_array[n, 3]

        temp_array = loc_array_tracklet_colored[loc_array_tracklet_colored[:, 6] == temp_tracklet_id, :]
        temp_array[:, 7] = temp_tracklet_alpha * np.ones_like(temp_array[:, 7])
        temp_array[:, 8] = temp_tracklet_D * np.ones_like(temp_array[:, 8])

        loc_array_tracklet_colored[loc_array_tracklet_colored[:, 6] == temp_tracklet_id, :] = temp_array

    if path:

        header_names = ['track id', 'time in s', 'efo in hz', 'x in um', 'y in um', 'z in um', 'tracklet id',
                        'tracklet alpha', 'tracklet D in um^2/s']
        loc_array_tracklet_colored_with_header = np.vstack([header_names, loc_array_tracklet_colored])

        np.savetxt(path + '/loc_array_tracklet_colored.csv',
                   loc_array_tracklet_colored_with_header, delimiter=',', fmt="%s")

    return loc_array_tracklet_colored



'''
Tracks plotter function:
    -plots tracks of all tracklets
    -inputs are are loc table with tracklets (output of tracklet_splitter), array with the diffusion parameters of
    tracklets (diffusian_array, 2nd outout of MSD calculator function), maximum number of tracklets plotted 
    (tracklet_max), parameter that is used to color the tracklets (color_by, has to be either "D" or "alpha"), 
    minimum for colormap (min_color),maximum for colormap (max_color), name of the colormap used (colormap_name), 
    color for part of the tracks that diffusion parameters could not be calculated for (nan_color), opacity of nan_color
    (nan_alpha), which font axis label uses (font_family), which font size for axis label (font_size), 
    camera angle (elev), camera angle (azim), camera angle (roll), if path is specified output is saved there as .pdf 
    (once with and once without gridlines)
    -output is the plot of the tracklets
'''
def tracks_plotter(loc_array_tracklet, diffusion_array, tracklet_max, line_width, color_by, min_color, max_color, colormap_name,
                   nan_color, nan_alpha, font_family, font_size, elev, azim, roll, path):

    if color_by not in ["D", "alpha"]:
        raise ValueError("Invalid color_by! Allowed options are: 'D', 'alpha'")

    tracklet_list = np.unique(loc_array_tracklet[:, 6])
    tracklet_list_not_nan = np.unique(diffusion_array[:, 0])

    if color_by == 'D':
        color_bar_label = r'D in $\mathrm{µm^2}$s$^{-1}$'
        column_for_color = diffusion_array[:, 3]

    elif color_by == 'alpha':
        color_bar_label = r'alpha'
        column_for_color = diffusion_array[:, 2]

    column_for_color[column_for_color > max_color] = max_color
    column_for_color[column_for_color < min_color] = min_color
    column_for_color_normalized = (column_for_color - min_color) / (max_color - min_color)

    # Create a colormap
    colormap_blank = plt.get_cmap(colormap_name)
    colormap = colormap_blank(column_for_color_normalized)

    #Change font and fontsize
    plt.rcParams['font.family'] = font_family
    plt.rcParams['font.size'] = font_size

    # Increase the size of the graph area by adjusting subplot parameters
    fig100 = plt.figure(100, figsize=(15, 10))  # Increase the overall figure size
    ax = fig100.add_subplot(111, projection='3d', position = [0, 0, 1, 1])  # Adjust width and position of main plot

    ax.view_init(elev=elev, azim=azim, roll=roll)
    ax.set_box_aspect((1, 1, 1))

    max_x = np.max(loc_array_tracklet[:, 3])
    max_y = np.max(loc_array_tracklet[:, 4])
    max_z = np.max(loc_array_tracklet[:, 5])
    max_dim = (np.max(np.array([max_x, max_y, max_z])))
    ax.set_xlim([0, max_dim])
    ax.set_ylim([0, max_dim])
    ax.set_zlim([0, max_dim])

    if tracklet_max < tracklet_list.shape[0]:
        tracklet_range = tracklet_max
    else:
        tracklet_range = tracklet_list.shape[0]

    for n in range(tracklet_range):

        temp_tracklet_id = tracklet_list[n]
        temp_array = loc_array_tracklet[loc_array_tracklet[:, 6] == temp_tracklet_id]

        if n + 1 < tracklet_list.shape[0]:
            temp_tracklet_id_next = tracklet_list[n+1]
            temp_array_next = loc_array_tracklet[loc_array_tracklet[:, 6] == temp_tracklet_id_next]

            # conncect all tracklets that lie within one track
            if temp_array[-1, 0] == temp_array_next[0, 0]:

                x_temp = np.array([temp_array[-1, 3], temp_array_next[0, 3]])
                y_temp = np.array([temp_array[-1, 4], temp_array_next[0, 4]])
                z_temp = np.array([temp_array[-1, 5], temp_array_next[0, 5]])

                plt.plot(x_temp, y_temp, z_temp, c=nan_color, alpha=nan_alpha, linewidth=line_width)

        if np.isin(temp_tracklet_id, tracklet_list_not_nan):

            temp_color = colormap[np.where(diffusion_array[:, 0] == temp_tracklet_id)]
            plt.plot(temp_array[:, 3], temp_array[:, 4], temp_array[:, 5], c=temp_color, linewidth=line_width)

        else:

            plt.plot(temp_array[:, 3], temp_array[:, 4], temp_array[:, 5], c=nan_color, alpha=nan_alpha, linewidth=line_width)


    # Add colorbar
    norm = plt.Normalize(min_color, max_color)
    sm = cm.ScalarMappable(cmap=colormap_name, norm=norm)
    sm.set_array([])  # Dummy empty array to set the range
    ax2 = fig100.add_subplot(212)  # Adjust width and position of main plot
    ax2.axis('off')
    cbar = plt.colorbar(sm, ax=ax2, orientation='horizontal', shrink=0.5)
    cbar.set_label(color_bar_label)


    if path:

        fig100.savefig(path + '/tracks' + '_grid' + '.pdf', bbox_inches="tight")
        ax.axis('off')
        fig100.savefig(path + '/tracks' + '.pdf', bbox_inches="tight")

    return fig100



'''
Single track plotter function:
    -does the same as track plotter function but for a single track and it's constituent tracklets instead of all 
    tracklets
    -inputs are are loc table with tracklets (output of tracklet_splitter), array with the diffusion parameters of
    tracklets (diffusian_array, 2nd outout of MSD calculator function), track id for which tracklets are plotted 
    (tracklet_nr), parameter that is used to color the tracklets (color_by, has to be either "D" or "alpha"), 
    minimum for colormap (min_color),maximum for colormap (max_color), name of the colormap used (colormap_name), 
    color for part of the tracks that diffusion parameters could not be calculated for (nan_color), opacity of nan_color
    (nan_alpha), which font axis label uses (font_family), which font size for axis label (font_size), 
    camera angle (elev), camera angle (azim), camera angle (roll), if path is specified output is saved there as .pdf
    (once with and once without gridlines)
    -output is the plot of the tracklets
'''
def single_track_plotter(loc_array_tracklet, diffusion_array, tracklet_nr, line_width, color_by, min_color, max_color, colormap_name,
                   nan_color, nan_alpha, font_family, font_size, elev, azim, roll, path):

    if color_by not in ["D", "alpha"]:
        raise ValueError("Invalid color_by! Allowed options are: 'D', 'alpha'")

    track_indices = np.where(loc_array_tracklet[:, 0] == tracklet_nr);
    range_min = np.min(track_indices)
    range_max = np.max(track_indices)

    loc_array_tracklet = loc_array_tracklet[range_min:range_max, :]
    loc_array_tracklet[:, 3] = loc_array_tracklet[:, 3] - np.min(loc_array_tracklet[:, 3])
    loc_array_tracklet[:, 4] = loc_array_tracklet[:, 4] - np.min(loc_array_tracklet[:, 4])
    loc_array_tracklet[:, 5] = loc_array_tracklet[:, 5] - np.min(loc_array_tracklet[:, 5])

    tracklet_list = np.unique(loc_array_tracklet[:, 6])
    tracklet_list_not_nan = np.unique(diffusion_array[:, 0])

    if color_by == 'D':
        color_bar_label = r'D in $\mathrm{µm^2}$s$^{-1}$'
        column_for_color = diffusion_array[:, 3]

    elif color_by == 'alpha':
        color_bar_label = r'alpha'
        column_for_color = diffusion_array[:, 2]

    column_for_color[column_for_color > max_color] = max_color
    column_for_color[column_for_color < min_color] = min_color
    column_for_color_normalized = (column_for_color - min_color) / (max_color - min_color)

    # Create a colormap
    colormap_blank = plt.get_cmap(colormap_name)
    colormap = colormap_blank(column_for_color_normalized)

    #Change font and fontsize
    plt.rcParams['font.family'] = font_family
    plt.rcParams['font.size'] = font_size

    #Increase the size of the graph area by adjusting subplot parameters
    fig101 = plt.figure(101, figsize=(15, 10))  # Increase the overall figure size
    ax = fig101.add_subplot(111, projection='3d', position = [0, 0, 1, 1])  # Adjust width and position of main plot

    ax.view_init(elev=elev, azim=azim, roll=roll)
    ax.set_box_aspect((1, 1, 1))

    max_x = np.max(loc_array_tracklet[:, 3])
    max_y = np.max(loc_array_tracklet[:, 4])
    max_z = np.max(loc_array_tracklet[:, 5])
    max_dim = (np.max(np.array([max_x, max_y, max_z])))
    ax.set_xlim([0, max_dim])
    ax.set_ylim([0, max_dim])
    ax.set_zlim([0, max_dim])

    for n in range(tracklet_list.shape[0]):

        temp_tracklet_id = tracklet_list[n]
        temp_array = loc_array_tracklet[loc_array_tracklet[:, 6] == temp_tracklet_id]

        if n + 1 < tracklet_list.shape[0]:
            temp_tracklet_id_next = tracklet_list[n+1]
            temp_array_next = loc_array_tracklet[loc_array_tracklet[:, 6] == temp_tracklet_id_next]

            # conncect all tracklets that lie within one track
            if temp_array[-1, 0] == temp_array_next[0, 0]:

                x_temp = np.array([temp_array[-1, 3], temp_array_next[0, 3]])
                y_temp = np.array([temp_array[-1, 4], temp_array_next[0, 4]])
                z_temp = np.array([temp_array[-1, 5], temp_array_next[0, 5]])

                plt.plot(x_temp, y_temp, z_temp, c=nan_color, alpha=nan_alpha, linewidth=line_width)

        if np.isin(temp_tracklet_id, tracklet_list_not_nan):
        #if np.isin(temp_tracklet_id, tracklet_list_not_nan) and n >= 68 and n <= 525:

            temp_color = colormap[np.where(diffusion_array[:, 0] == temp_tracklet_id)]
            plt.plot(temp_array[:, 3], temp_array[:, 4], temp_array[:, 5], c=temp_color, linewidth=line_width)

        else:

            plt.plot(temp_array[:, 3], temp_array[:, 4], temp_array[:, 5], c=nan_color, alpha=nan_alpha, linewidth=line_width)


    # Add colorbar
    norm = plt.Normalize(min_color, max_color)
    sm = cm.ScalarMappable(cmap=colormap_name, norm=norm)
    sm.set_array([])  # Dummy empty array to set the range
    ax2 = fig101.add_subplot(212)  # Adjust width and position of main plot
    ax2.axis('off')
    cbar = plt.colorbar(sm, ax=ax2, orientation='horizontal', shrink=0.5)
    cbar.set_label(color_bar_label)

    if path:

        fig101.savefig(path + '/single_track_' + str(tracklet_nr) + '_grid' + '.pdf', bbox_inches="tight")
        ax.axis('off')
        fig101.savefig(path + '/single_track_' + str(tracklet_nr) + '.pdf', bbox_inches="tight")

    return fig101



'''
Hist plotter function:
    -make histograms for all tracklets
    -inputs are array with the msds of all tracklets (msd_all, 1st output of MSD calculator function), the diffusion 
    parameters of tracklets (diffusioan_array, 2nd outout of MSD calculator function), maximum number of tracklets 
    plotted (tracklet_max), which font axis label uses (font_family), which font size for axis label (font_size), 
    camera angle (elev), camera angle (azim), camera angle (roll), if path is specified output is saved there as .pdf
    -outputs are the following plots in order: time-lag, MSD-lag, MSD-time, log(MSD)-log(time), hist of alpha, 
    hist of D, hist of length, hist of time, hist of timeerror
'''
def hist_plotter(msd_all, diffusion_array, tracklet_max, font_family, font_size, path=''):

    max_lag = int(np.max(msd_all[:, 1]))
    tracklet_list = np.unique(diffusion_array[:, 0])

    fig1 = None
    fig2 = None
    fig3 = None
    fig4 = None
    fig5 = None
    fig6 = None
    fig7 = None
    fig8 = None
    fig9 = None

    # Change font and fontsize
    plt.rcParams['font.family'] = font_family
    plt.rcParams['font.size'] = font_size

    if tracklet_max < tracklet_list.shape[0]:
        tracklet_range = tracklet_max
    else:
        tracklet_range = tracklet_list.shape[0]

    for n in range(tracklet_range):

        # time-lag
        fig1 = plt.figure(1)
        temp_x = msd_all[n * max_lag:n * max_lag + max_lag, 1]
        temp_y = msd_all[n * max_lag:n * max_lag + max_lag, 4]
        plt.plot(temp_x, temp_y)
        plt.title('time-lag')
        plt.xlabel('lag')
        plt.ylabel('time [s]')

        # MSD-lag
        fig2 = plt.figure(2)
        temp_x = msd_all[n * max_lag:n * max_lag + max_lag, 1]
        temp_y = msd_all[n * max_lag:n * max_lag + max_lag, 2]
        plt.plot(temp_x, temp_y)
        plt.title('MSD-lag')
        plt.xlabel('lag')
        plt.ylabel('MSD [$\mathrm{µm^2}$]')

        # MSD-time
        fig3 = plt.figure(3)
        temp_x = msd_all[n * max_lag:n * max_lag + max_lag, 4]
        temp_y = msd_all[n * max_lag:n * max_lag + max_lag, 2]
        plt.plot(temp_x, temp_y)
        plt.title('MSD-time')
        plt.xlabel('time in s')
        plt.ylabel('MSD [$\mathrm{µm^2}$]')

        # log10(MSD)-log10(time)
        fig4 = plt.figure(4)
        temp_x = np.log10(msd_all[n * max_lag:n * max_lag + 3, 4])
        temp_y = np.log10(msd_all[n * max_lag:n * max_lag + 3, 2])
        plt.plot(temp_x, temp_y)
        plt.title('log(MSD)-log(time)')
        plt.xlabel('log10(time [s])')
        plt.ylabel('log10(MSD [$\mathrm{µm^2}$])')

    n_bins = int(np.round(np.sqrt(diffusion_array[:, 4].shape[0])))

    #hist_alpha
    fig5 = plt.figure(5)
    plt.hist(diffusion_array[:, 2], bins=n_bins)
    plt.title('count-alpha')
    plt.xlabel('alpha')
    plt.ylabel('count')

    #hist_log10(D)
    fig6 = plt.figure(6)
    plt.hist(diffusion_array[:, 4], bins=n_bins)
    plt.title('count-log(D)')
    plt.xlabel('log10(D [$\mathrm{µm^2}$s$^{-1}$])')
    plt.ylabel('count')

    #hist_length
    fig7 = plt.figure(7)
    plt.hist(diffusion_array[:, 1], bins=n_bins)
    plt.title('count-length')
    plt.xlabel('length (locnumber)')
    plt.ylabel('count')

    # hist_time
    fig8 = plt.figure(8)
    plt.hist(diffusion_array[:, 5], bins=n_bins)
    plt.title('count-time')
    plt.xlabel('time [s]')
    plt.ylabel('count')

    # hist_timeerror
    fig9 = plt.figure(9)
    plt.hist(diffusion_array[:, 6], bins=n_bins)
    plt.title('count-time error')
    plt.xlabel('time error [%]')
    plt.ylabel('count')

    fig_list = [fig1, fig2, fig3, fig4, fig5, fig6, fig7, fig8, fig9]

    if path:

        for n in fig_list:
            n.savefig(path + '/plot_' + str(n.axes[0].get_title()) + '.pdf', bbox_inches = "tight")

    return fig_list

