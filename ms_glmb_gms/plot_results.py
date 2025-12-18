import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import LinearSegmentedColormap

def plot_results(model, truth, meas, est, title, filename):
    X_track, k_birth, k_death = extract_tracks(truth.X, truth.track_list, truth.total_tracks)

    labelcount = countestlabels(meas, est)
    ca_obj = ca()
    colorarray = ca_obj.makecolorarray(labelcount)
    est.total_tracks = labelcount
    est.track_list = {key: np.empty(0, dtype=int) for key in range(0, truth.K)};
    for k in range(0, truth.K):
        for eidx in range(0, est.X[k].shape[1]):
            est.track_list[k] = np.append(est.track_list[k], assigncolor(est.L[k][:, eidx], colorarray))

    Y_track, l_birth, l_death = extract_tracks(est.X, est.track_list, est.total_tracks)

    # plot ground truths
    # limit = np.array([model.range_c[0, 0, 0], model.range_c[0, 0, 1], model.range_c[1, 0, 0], model.range_c[1, 0, 1],
    #                   model.range_c[2, 0, 0], model.range_c[2, 0, 1]])
    
    # Generate a colormap for different tracks
    colors = cm.get_cmap('tab10', max(truth.total_tracks, est.total_tracks))
    plt.figure(figsize=(10, 6), dpi=200)
    ax = plt.axes(projection='3d', computed_zorder=False)

    # Hide z-axis completely
    ax.set_zticks([])                 # no ticks
    ax.set_zticklabels([])            # no tick labels
    ax.zaxis.line.set_visible(False)  # hide the z-axis line
    ax.zaxis.pane.set_visible(False)  # hide the background pane

    ax.view_init(elev=90, azim=-90) # top down viewing angle
    legend_elements = []     # Lists to store legend elements
    
    # Define RGB values for middle grey and dark grey
    dark_grey = (0.05, 0.05, 0.05)
    light_grey = (0.85, 0.85, 0.85)
    

    # Create a custom colormap from middle grey to dark grey using RGB values
    cmap_grey_to_dark = LinearSegmentedColormap.from_list(
        'grey_to_dark', 
        [(0, dark_grey), (1, light_grey)]  # Mapping 0 to middle grey and 1 to dark grey
    )

    # Plot measurements in 3D
    for s in range(model.N_sensors):
        for k in range(0, meas.K):
            if meas.Z[(k, s)].size == 0:
                continue
            # # Plot measurements as points (scatter plot)
            # ax.scatter(meas.Z[(k, s)][0, :], meas.Z[(k, s)][1, :], meas.Z[(k, s)][2, :], marker='x', s=50,
            #         color=0.7 * np.ones((1, 3)), label=f'Measurements for Track {k}')
                    # Normalize the timestamp k (between 0 and K-1) to the range [0, 1]
            norm_k = k / (meas.K - 1)  # Normalized value between 0 (dark grey) and 1 (light grey)
            
            # Use a colormap (e.g., gray) to map the normalized timestamp to a color
            color = cmap_grey_to_dark(norm_k)  # cm.gray generates a grey color based on the normalized value

            if s == 0:
                marker = 'x' # video detections
                # marker = '+'  # audio detections
                size = 30

            elif s == 1:
                marker = '+'
                size = 40
        
            # Plot measurements as points (scatter plot)
            ax.scatter(
                meas.Z[(k, s)][0, :], 
                meas.Z[(k, s)][1, :], 
                meas.Z[(k, s)][2, :], 
                marker=marker, 
                s=size, 
                color=color, 
                label=f'Measurements for Track {k}' if k == 0 else "",  # Avoid repeated labels
                alpha=1.0,
                zorder=1
            )

    


    for i in range(truth.total_tracks):
        Pt = X_track[:, np.arange(k_birth[i], k_death[i], 1), i]
        Pt = Pt[[0, 2, 4], :]
        
        # Plot track
        # ax.plot(Pt[0, :], Pt[1, :], Pt[2, :], color=colors(i), label=f'Track ID {i+1}', alpha=0.75)
        ax.plot(Pt[0, :], Pt[1, :], Pt[2, :], color='black', label=f'Track ID {i+1}', alpha=1.0, zorder=7)
        
        # Plot birth marker
        # ax.scatter(Pt[0, 0], Pt[1, 0], Pt[2, 0], color=colors(i), marker='o', s=60, alpha=0.75)
        ax.scatter(Pt[0, 0], Pt[1, 0], Pt[2, 0], color='black', marker='o', s=60, alpha=1.0, zorder=8)
        
        # Plot death marker
        # ax.scatter(Pt[0, -1], Pt[1, -1], Pt[2, -1], color=colors(i), marker='^', s=60, alpha=0.75)
        ax.scatter(Pt[0, -1], Pt[1, -1], Pt[2, -1], color='black', marker='^', s=60, alpha=1.0, zorder=9)

    # Add birth and death markers once to the legend
    legend_elements.append(plt.Line2D([0], [0], marker='o', color='black', markersize=8, linestyle='None', label='Object Birth', alpha=1.0))
    legend_elements.append(plt.Line2D([0], [0], marker='^', color='black', markersize=8, linestyle='None', label='Object Death', alpha=1.0))

    # Add track IDs to the legend
    legend_elements.append(plt.Line2D([0], [0], color='black', lw=2, label=f'True Tracks', alpha=1.0))
    # for i in range(truth.total_tracks):
    #     # legend_elements.append(plt.Line2D([0], [0], color=colors(i), lw=2, label=f'True Track ID {i+1}', alpha=0.5))
    #     legend_elements.append(plt.Line2D([0], [0], color='black', lw=2, label=f'True Track ID {i+1}', alpha=0.75))

    # Add measurements to the legend once for each sensor
    for s in range(model.N_sensors):
        if s == 0:
            # legend_elements.append(plt.Line2D([0], [0], marker='+', color='black', markersize=8, linestyle='None', label='Audio Detections'))
            legend_elements.append(plt.Line2D([0], [0], marker='x', color='black', markersize=8, linestyle='None', label='Video Detections'))
        elif s == 1:
            legend_elements.append(plt.Line2D([0], [0], marker='+', color='black', markersize=8, linestyle='None', label='Audio Detections'))

    # Add a colorbar to represent the time-based color coding
    sm = plt.cm.ScalarMappable(cmap=cmap_grey_to_dark, norm=plt.Normalize(vmin=0, vmax=(meas.K - 1)/10)) # /10 to account for 0.1s timesteps instead of 1
    sm.set_array([])  # Required for the colorbar to work
    cbar = plt.colorbar(sm, ax=ax, shrink=0.7, pad=0)
    cbar.set_label('Time [s]')

    # Add the custom legend for the time-based color coding
    # legend_elements.append(plt.Line2D([0], [0], color=cm.viridis(0), lw=2, label='Start Time [s]', alpha=0.7))
    # legend_elements.append(plt.Line2D([0], [0], color=cm.viridis(1), lw=2, label='End Time [s]', alpha=0.7))

    # Plot estimated tracks and markers in 3D
    for i in range(est.total_tracks):
        Pt = Y_track[:, np.arange(l_birth[i], l_death[i], 1), i]
        Pt = Pt[[0, 2, 4], :]

        # Print trajectory points
        # print(Pt.T)  # Transpose to one row per timestep: [x, y, z]

        # Plot track
        ax.plot(Pt[0, :], Pt[1, :], Pt[2, :], color=colors(i), label=f'Track ID {i+1}', alpha=1.0, zorder=10)
        
        # Find the first valid (non-NaN) 3D point (birth)
        for j in range(Pt.shape[1]):
            if not np.any(np.isnan(Pt[:, j])):
                birth_point = Pt[:, j]
                break

        # Find the last valid (non-NaN) 3D point (death)
        for j in range(Pt.shape[1] - 1, -1, -1):
            if not np.any(np.isnan(Pt[:, j])):
                death_point = Pt[:, j]
                break

        # Plot the birth marker at the first valid 3D point
        print(f"birth:{birth_point[0], birth_point[1], birth_point[2]}")
        ax.scatter(birth_point[0], birth_point[1], birth_point[2], color=colors(i), marker='o', s=60, zorder=11)
        
        # Plot the death marker at the last valid 3D point
        print(f"death:{death_point[0], death_point[1], death_point[2]}")
        ax.scatter(death_point[0], death_point[1], death_point[2], color=colors(i), marker='^', s=60, zorder=11)

    # # Add estimated birth and death markers once to the legend
    # legend_elements.append(plt.Line2D([0], [0], marker='o', color='black', markersize=8, linestyle='None', label='Estimated Birth'))
    # legend_elements.append(plt.Line2D([0], [0], marker='^', color='black', markersize=8, linestyle='None', label='Estimated Death'))

    print(f"est.total_tracks: {est.total_tracks}")
    # Add estimated track IDs to the legend
    for i in range(est.total_tracks):
        print(i)
        legend_elements.append(plt.Line2D([0], [0], color=colors(i), lw=2, label=f'Estimated Track ID {i+1}'))



    # Set axis labels
    ax.set_xlabel('x [m]')
    ax.set_ylabel('y [m]')
    ax.set_zlabel('z [m]')
    ax.set_zlabel('')                 # hide the axis label

    # Set axis limits
    # ax.set_xlim3d(limit[0], limit[1])
    # ax.set_ylim3d(limit[2], limit[3])
    # ax.set_zlim3d(limit[4], limit[5])
    ax.set_xlim3d(0, 5)
    # ax.set_xlim3d(-1, 11)
    ax.set_ylim3d(0, 5)
    ax.set_zlim3d(0, 3)


    plt.title(f"MS-GLMB Multi-object Tracking Results - {title}")
    # ax.legend(handles=legend_elements, loc='upper left', fontsize='small', bbox_to_anchor=(0.9, 1.0)) # bbox_to_anchor to move legend relative to loc
    ax.legend(handles=legend_elements, loc='center left', fontsize='medium', bbox_to_anchor=(-0.4, 0.5)) # bbox_to_anchor to move legend relative to loc
    plt.savefig(f"{filename}.png")



    # # Plot each axis for each sensor separately
    # for s in range(model.N_sensors):
    #     # plot tracks and measurements in x/y
    #     # plot x measurement
    #     fig, (axs1, axs2, axs3) = plt.subplots(3)
        
    #     for k in range(0, meas.K):
    #         if meas.Z[(k, s)].size == 0:
    #             continue
    #         plt_x_meas = axs1.scatter(k * np.ones((meas.Z[(k, s)].shape[1], 1)), meas.Z[(k, s)][0, :], marker='x', s=50,
    #                                   color=0.7 * np.ones((1, 3)), label='Measurements')
    #         plt_y_meas = axs2.scatter(k * np.ones((meas.Z[(k, s)].shape[1], 1)), meas.Z[(k, s)][1, :], marker='x', s=50,
    #                                   color=0.7 * np.ones((1, 3)), label='Measurements')
    #         plt_z_meas = axs3.scatter(k * np.ones((meas.Z[(k, s)].shape[1], 1)), meas.Z[(k, s)][2, :], marker='x', s=50,
    #                                   color=0.7 * np.ones((1, 3)), label='Measurements')
    #     # plot x, y, z track
    #     for i in range(0, truth.total_tracks):
    #         P = X_track[:, np.arange(k_birth[i], k_death[i], 1), i]
    #         P = P[[0, 2, 4], :]
    #         plt_x_truth = axs1.plot(np.arange(k_birth[i], k_death[i], 1), P[0, :], linestyle='-', linewidth=1,
    #                                 color=0 * np.ones((1, 3)), label='True tracks')
    #         plt_y_truth = axs2.plot(np.arange(k_birth[i], k_death[i], 1), P[1, :], linestyle='-', linewidth=1,
    #                                 color=0 * np.ones((1, 3)), label='True tracks')
    #         plt_z_truth = axs3.plot(np.arange(k_birth[i], k_death[i], 1), P[2, :], linestyle='-', linewidth=1,
    #                                 color=0 * np.ones((1, 3)), label='True tracks')
    #     # plt.show()
        
    #     # plot x, y, z estimate
    #     # Initialize empty lists to hold the plot objects
    #     # plt_x_est, plt_y_est, plt_z_est = [], [], []
    #     for k in range(meas.K):
    #         if len(est.X[k]) == 0:
    #             continue
    #         P = est.X[k][[0, 2, 4]]
    #         L = est.L[k]
    #         for eidx in range(P.shape[1]):
    #             color = assigncolor(L[:, eidx], colorarray)[0]
    #             plt_x_est = axs1.plot(k, P[0, eidx], marker='.', color=colors(eidx), label='Estimates')
    #             plt_y_est = axs2.plot(k, P[1, eidx], marker='.', color=colors(eidx), label='Estimates')
    #             plt_z_est = axs3.plot(k, P[2, eidx], marker='.', color=colors(eidx), label='Estimates')
    #     axs1.legend(handles=[plt_x_meas, plt_x_truth[0], plt_x_est[0]], loc='upper left')
    #     axs1.set_xlabel('Time')
    #     axs1.set_ylabel('x-coordinate (m)')
    #     axs2.set_xlabel('Time');
    #     axs2.set_ylabel('y-coordinate (m)')
    #     axs3.set_xlabel('Time');
    #     axs3.set_ylabel('z-coordinate (m)')
    #     fig.suptitle("Sensor Measurements, Ground Truth, and Estimates Over Time")
    plt.show()


def plot_truth_meas(model, truth, meas):
    X_track, k_birth, k_death = extract_tracks(truth.X, truth.track_list, truth.total_tracks)

    # plot ground truths
    limit = np.array([model.range_c[0, 0], model.range_c[0, 1], model.range_c[1, 0], model.range_c[1, 1]])
    plt.figure(figsize=(9, 3))
    for i in range(0, truth.total_tracks):
        Pt = X_track[:, np.arange(k_birth[i], k_death[i], 1), i]
        Pt = Pt[[0, 2], :]
        plt.plot(Pt[0, :], Pt[1, :], 'k-');
        plt.plot(Pt[0, 0], Pt[1, 0], 'ko', 6);
        plt.plot(Pt[0, (k_death[i] - k_birth[i] - 1)], Pt[1, (k_death[i] - k_birth[i] - 1)], 'k^', 6)
    plt.axis(limit)
    plt.title('Ground Truths')
    # plt.show()

    # plot tracks and measurements in x/y
    # plot x measurement
    fig, (axs1, axs2) = plt.subplots(2, figsize=(12, 12))
    for k in range(0, meas.K):
        if meas.Z[k].size != 0:
            plt_x_meas = axs1.scatter(k * np.ones((meas.Z[k].shape[1], 1)), meas.Z[k][0, :], marker='x',
                     s=50, color=0.7 * np.ones((1, 3)), label='Measurements')
    # plot x track
    for i in range(0, truth.total_tracks):
        Px = X_track[:, np.arange(k_birth[i], k_death[i], 1), i]
        Px = Px[[0, 2], :]
        plt_x_truth = axs1.plot(np.arange(k_birth[i], k_death[i], 1), Px[0, :], linestyle='-', linewidth=1,
                 color=0 * np.ones(3), label='True tracks')
    axs1.legend(handles=[plt_x_meas, plt_x_truth[0]])
    axs1.set_xlabel('Time')
    axs1.set_ylabel('x-coordinate (m)')

    # plot y measurement
    # plt.subplot(212)
    for k in range(0, meas.K):
        if meas.Z[k].size != 0:
            plt_y_meas = axs2.scatter(k * np.ones((meas.Z[k].shape[1], 1)), meas.Z[k][1, :], marker='x',
                     s=50, color=0.7 * np.ones((1, 3)), label='Measurements')
    # plot y track
    for i in range(0, truth.total_tracks):
        Py = X_track[:, np.arange(k_birth[i], k_death[i], 1), i]
        Py = Py[[0, 2], :]
        plt_y_truth = axs2.plot(np.arange(k_birth[i], k_death[i], 1), Py[1, :], linestyle='-', linewidth=1,
                 color=0 * np.ones(3), label='True tracks')
    # plt.legend('Estimates')
    axs2.set_xlabel('Time');
    axs2.set_ylabel('y-coordinate (m)')
    axs2.legend(handles=[plt_y_meas, plt_y_truth[0]])
    # plt.show()

    return fig, axs1, axs2

class ca:
    def makecolorarray(self, nlabels):
        lower = 0.1
        upper = 0.9
        rrr = np.random.rand(1, nlabels) * (upper - lower) + lower
        ggg = np.random.rand(1, nlabels) * (upper - lower) + lower
        bbb = np.random.rand(1, nlabels) * (upper - lower) + lower
        self.rgb = np.concatenate((rrr, ggg, bbb))
        self.lab = np.empty((nlabels), dtype=object)
        self.cnt = -1

        return self


def assigncolor(label, colorarray):
    str = np.array2string(label, separator='*')[1:-1] + '*'
    tmp = (str == colorarray.lab)
    if np.nonzero(tmp)[0].size > 0:
        idx = np.nonzero(tmp)[0]
    else:
        colorarray.cnt = colorarray.cnt + 1;
        colorarray.lab[colorarray.cnt] = str
        idx = colorarray.cnt

    return idx


def countestlabels(meas, est):
    labelstack = np.empty((2, 0), dtype=int)
    for k in range(0, meas.K):
        labelstack = np.concatenate((labelstack, est.L[k]), axis=1)
    c, _, _ = np.unique(labelstack, return_index=True, return_inverse=True, axis=1)
    count = c.shape[1]
    return count


def extract_tracks(X, track_list, total_tracks):
    K = len(X)
    k = K - 1
    x_dim = X[k].shape[0]
    while x_dim == 0:
        x_dim = X[k].shape[0]
        k = k - 1
    X_track = np.zeros((x_dim, K, total_tracks))
    X_track[:] = np.NaN
    k_birth = np.zeros((total_tracks, 1), dtype=int)
    k_death = np.zeros((total_tracks, 1), dtype=int)

    max_idx = 0;
    for k in range(0, K):
        if X[k].size != 0:
            X_track[:, k, track_list[k]] = X[k]
        else:
            continue
        if max(track_list[k]) > max_idx:  # new target born?
            idx = np.nonzero(track_list[k] > max_idx)[0]
            k_birth[track_list[k][idx]] = k

        max_idx = max(track_list[k])
        k_death[track_list[k]] = k + 1

    return X_track, k_birth, k_death


def get_comps(X, c):
    if len(X) == 0:
        Xc = {};
    else:
        Xc = X[c, :]
    return Xc

def write_results_jsonl(model, est, filename='tracking.jsonl'):
    """
    Write tracking results from `est` into a JSONL file (one JSON object per time step).
    Each line has the form:
    {"timestamp":"<seconds with 16 fractional digits>","objects":[{"object_id":0,"position":[x,y,z]}, ...]}

    - timestamp is formatted as a string with 16 fractional digits (e.g. "0.1000000000000000").
    - position is taken from state vector indices [0,2,4] (i.e. x,y,z).
    - model.T is used to convert discrete time k -> seconds: t = k * model.T
    """
    # Try to infer total_tracks if not present
    if hasattr(est, 'total_tracks') and getattr(est, 'total_tracks', None) is not None:
        total_tracks = int(est.total_tracks)
    else:
        total_tracks = -1
        if hasattr(est, 'track_list'):
            for k in est.track_list:
                if est.track_list[k].size > 0:
                    total_tracks = max(total_tracks, int(np.max(est.track_list[k])))
        if total_tracks == -1:
            # fallback: no track_list found or all empty -> assume zero tracks
            total_tracks = 0
        else:
            total_tracks = total_tracks + 1

    # If est.track_list missing or malformed, try a safe default for extract_tracks call:
    # extract_tracks expects track_list to be a mapping from time index -> array of track indices.
    if not hasattr(est, 'track_list'):
        raise AttributeError("est.track_list is required by write_results_jsonl but was not found.")

    # Build the dense track tensor Y_track with lifetimes using your extract_tracks helper
    Y_track, l_birth, l_death = extract_tracks(est.X, est.track_list, int(total_tracks))

    # Number of time steps (K)
    K = Y_track.shape[1]

    # Open file and write JSONL lines
    with open(filename, 'w') as fh:
        for k in range(K):
            t_seconds = k * float(getattr(model, 'T', 1.0))
            # format timestamp as string with 16 fractional digits
            timestamp_str = f"{t_seconds:.16f}"

            objects_strs = []
            for obj_id in range(int(total_tracks)):
                # extract x,y,z from state vector
                pos = Y_track[[0, 2, 4], k, obj_id]
                # skip tracks that are not present at this time (all NaN)
                if np.all(np.isnan(pos)):
                    continue

                # format numeric components with 16 fractional digits (produce JSON numbers)
                x_s = f"{float(pos[0]):.16f}"
                y_s = f"{float(pos[1]):.16f}"
                z_s = f"{float(pos[2]):.16f}"

                obj_str = f'{{"object_id":{obj_id},"position":[{x_s},{y_s},{z_s}]}}'
                objects_strs.append(obj_str)

            objects_joined = ",".join(objects_strs)
            line = f'{{"timestamp":"{timestamp_str}","objects":[{objects_joined}]}}'
            fh.write(line + "\n")

    return filename