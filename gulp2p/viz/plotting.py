# plotting.py
import logging
import numpy as np
import pandas as pd
import pingouin

import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Rectangle
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
# import matplotlib.lines as mlines
from skimage.measure import block_reduce
from scipy.signal import savgol_filter

# import unityvr.preproc.logproc as lp
from unityvr.analysis import posAnalysis
import unityvr.viz.utils as uvrvisutils
from gulp2p import preproc
import gulp2p.analysis.head_direction as hd

logger = logging.getLogger(__name__)

def plot_colorbar(cax, F_plot, F_lims, cbarlabel):
    """Plot the colorbar for the given F_plot

    Args:
        fig (~matplotlib.figure.Figure): Figure to add colorbar to
        cax (Axes): Axes to plot colorbar in
        F_plot (AxesImage): Plot of florescence
        F_lims (list): min and max florescence values
        cbarlabel (str): label of colorbar
    """
    # Plot colorbar
    fig = cax.get_figure()
    cbar = fig.colorbar(F_plot, cax=cax)
    if ((F_lims[0] > 0) and (F_lims[1] > 0)) or (F_lims[0] < 0) and (F_lims[1] < 0):
        # Doesn't pass through 0 (eg. raw florescence)
        ticks = [F_lims[0], F_lims[1]]
        cbar.set_label(cbarlabel, labelpad=-25)
    else:
        # Passes through 0 (eg. delta florescence)
        ticks = [F_lims[0], 0, F_lims[1]]
        cbar.set_label(cbarlabel, labelpad=-12)
    cbar.set_ticks(ticks)
    if (F_lims[0] > 10**2) or (F_lims[1] > 10**2):
        tick_labels = [f"{lim:g}" for lim in ticks]
    else:
        tick_labels = [f"{lim:.2g}" for lim in ticks]
    cbar.ax.set_yticklabels(tick_labels)

def plot_florescence(
        F, panel, cmap, aspect, fm_interval, F_lims, norm=None,
        withcbar=False, cax=None, cbarlabel=None,
        ):
    """Plot the florescence of F
    if with cbar is true, need to provide axes of cbar
    Note: Plotting fails if trial is too long, maximum length is 18 minutes

    Args:
        F (NDArray[float64]): ndarray of florescence, with shape (# of ROI's, # of frames)
        panel (Axes): Axes to draw plot in.
        cmap (str or Colormap): Colormap to use.
        aspect (float): Vertical to horizontal ratio of heatmap pixel, of form aspect:1.
        norm (TwoSlopeNorm): TwoSlopeNorm object with range of data.
        fm_interval (float): Time it takes to capture one frame (in seconds per frame).
        withcbar (bool, optional): Set to true to add a colorbar. Defaults to False.
        cbaraxes (Axes, optional): Axes to plot colorbar in. Defaults to None.
        cbarlabel (str, optional): label of colorbar. Defaults to None.
    """
    num_rois, num_frames = F.shape
    # Plot florescence
    extent = (0, num_frames * fm_interval,
              0, num_rois)
    F_plot = panel.imshow(
        F,
        cmap=cmap,
        interpolation="none",
        aspect=aspect,
        norm=norm,
        origin='lower',
        extent=extent
    )
    # panel.title.set_text(title)
    num_frames = F.shape[1]
    panel.set_ylabel("ROI")
    panel.set_yticks(
        [i+0.5 for i in range(num_rois) if i % 2 == 0],
        [i + 1 for i in range(num_rois) if i % 2 == 0],
    )
    # Plot colorbar
    if withcbar:
        plot_colorbar(cax, F_plot, F_lims, cbarlabel)

def removeJumps(dat, thresh):
    datNoJump = np.array(dat)

    datDiff = np.array(dat[:-1]) - np.array(dat[1:])
    for i,d in enumerate(datDiff):
        if abs(d) > thresh:
            datNoJump[i+1] = None

    return datNoJump   

def corrOverTime(x,y,time, pre_time, post_time):
    dt = np.round(np.diff(time).mean(),2)
    pre_time_pts = int(np.round(pre_time/dt))
    post_time_pts = int(np.round(post_time/dt))
    pt_range = np.arange(-pre_time_pts, post_time_pts)

    corr = pd.DataFrame({'time':[],'corr':[]})
    for pt in pt_range:
        corr_val = np.corrcoef(x[max(0,pt):min(len(x),len(x)+pt)],y[max(0,-pt):min(len(y),len(y)-pt)])[0,1]
        corr = pd.concat([corr, pd.DataFrame({'time':[dt*pt],'corr':[corr_val]})])

    return corr


###
# Functions added during refactor
###

## Plotting functions
def plot_deltaf_heatmap(expt, ax, mode="upper", vmin=None, vmax=None, with_int_hd=False, smooth=True):
    # Calculate length of all trials
    total_length = 0
    for trial in expt.trials:
        total_length += trial.length

    # Plot trials
    for index, trial in enumerate(expt.trials):

        # Select deltaf
        if mode == "full":
            deltaf = trial.synced_df.iloc[:, :trial.num_rois]
        if mode == "upper":
            deltaf = trial.synced_df.iloc[:, 8:trial.num_rois]
        if mode == "lower":
            deltaf = trial.synced_df.iloc[:, 0:8]


        # Set extent for trial period
        left = trial.synced_df['posTime'].iloc[0] - 0.5
        right = trial.synced_df['posTime'].iloc[0] + trial.length - 0.5
        bottom = - 0.5
        top = deltaf.shape[1] - 0.5
        extent = (left, right, bottom, top)

        # Savitzky-Golay filter
        if smooth:
            deltaf = savgol_filter(deltaf.T, window_length=11, polyorder=3).T

        # Plot df/f of trial
        ax.imshow(deltaf.T,
                  aspect="auto", interpolation="none",
                  vmin=vmin, vmax=vmax, extent=extent,
                  origin = 'lower')

        # Plot internal head direction
        if with_int_hd:
            internal_hd = hd.get_internal_head_direction(trial, mode='mean')

            # Change range from [0, 8) to [-0.5, 7.5), since that is the visual range on plots.
            mask = (internal_hd >= (expt.rois_per_arm - 0.5))
            internal_hd[mask] -= expt.rois_per_arm

            ax.plot(trial.synced_df['posTime'],
                    internal_hd,
                    marker='o', linestyle="", ms=0.5, color="C1", alpha=0.5)

        # Plot trial dividing line
        # if index != 0:
        #     ax.axvline(trial.synced_df['posTime'].iloc[0],
        #                linestyle="--",
        #                linewidth=1,
        #                color="gray",
        #                zorder=10)

    ax.set_xlim([0, expt.synced_df['posTime'].iloc[-1]])

def plot_external_head_direction(expt, ax):
    external_hd_rad = pingouin.convert_angles(expt.synced_df['angle'])
    ax.plot(expt.synced_df['posTime'],
            external_hd_rad,
            marker='o', linestyle="", ms=1, color="C1")

    for index, trial in enumerate(expt.trials):
        # Plot trial dividing line
        if index != 0:
            # Begining of trial
            ax.axvline(trial.synced_df['posTime'].iloc[0],
                       linestyle="--",
                       linewidth=1,
                       color="gray",
                       zorder=10)
        if index != (len(expt.trials) - 1):
            # End of trial
            ax.axvline(trial.synced_df['posTime'].iloc[-1],
                       linestyle="--",
                       linewidth=1,
                       color="gray",
                       zorder=10)

    ax.set_ylim([-np.pi, np.pi])
    ax.set_xlim([0, expt.synced_df['posTime'].iloc[-1]])

    yticks = np.linspace(-np.pi, np.pi, 5, endpoint=True)
    yticklabels = [r'-$\pi$', r'-$\pi/2$', '$0$', r'$\pi/2$', r'$\pi$']
    ax.set_yticks(yticks, yticklabels)

def plot_bump_profile(aligned_bumps, ax, with_std=True, with_fit=True, with_metric=True):
    # TODO: get aligned bumps from expt
    mean_bump = np.mean(aligned_bumps, axis=0)
    std_bump = np.std(aligned_bumps, axis=0)
    ax.plot(mean_bump, color='black', linewidth=2)
    ax.axhline(0, color='black', linestyle="--", alpha=0.5)

    if with_std:
        ax.fill_between(range(len(mean_bump)),
                        mean_bump + std_bump,
                        mean_bump - std_bump,
                        color="gray",
                        alpha=0.5,
                        zorder=1)

    if with_fit:
        cos_func = hd.standard_cos_func
        parameters, covariance, r_squared = hd.get_cos_fit(mean_bump, cos_func=cos_func)
        xdata = np.linspace(0, len(mean_bump)-1, 100)
        ydata = cos_func(xdata, *parameters)
        ax.plot(xdata, ydata)
    
    if with_metric:
         metric = f"R^2 = {r_squared:.3f}"
         ax.text(0.95, 0.95, metric, fontsize=7,
                 transform=ax.transAxes, ha='right', va='top')

    ax.set_xlabel("roi")
    ax.set_ylabel("df/f")

def plot_hd_offset_histogram(expt, ax, nbins=16, mode='best'):
    if mode == 'best':
        hd_offsets = {}
        for index, region in enumerate(['upper', 'lower', 'mean']):
            hd_offset = hd.get_hd_offset(expt, mode=region)
            circ_var = hd.get_circ_var(hd_offset)
            hd_offsets[region] = {'offset': hd_offset,
                                  'circ_var': circ_var}
        min_var_region = min(hd_offsets.keys(), key=lambda k: hd_offsets[k]['circ_var'])
        hd_offset = hd_offsets[min_var_region]['offset']
        circ_var = hd_offsets[min_var_region]['circ_var']

    else:
        hd_offset = hd.get_hd_offset(expt, mode=mode)

    ax.hist(hd_offset, bins=nbins)

    if mode != 'best':
        circ_var = hd.get_circ_var(hd_offset)
    metric = f"circ var = {circ_var:.3f}"
    ax.text(0.99, 0.975, metric, fontsize=9,
            transform=ax.transAxes, ha='right', va='top')

    ax.set_xlim([-np.pi, np.pi])
    ax.set_ylim([None, ax.get_ylim()[1]*1.1])

    xticks = np.linspace(-np.pi, np.pi, 5, endpoint=True)
    xticklabels = [r'-$\pi$', r'-$\pi/2$', '$0$', r'$\pi/2$', r'$\pi$']
    ax.set_xticks(xticks, xticklabels)

    if mode == 'best':
        region = min_var_region
    else:
        region = mode
    ax.set_xlabel(f"Offset Angle ({region} int. HD)")
    ax.set_ylabel("Counts")

def plot_expt(expt):
    # Plot experiment
    mosaic = [["upper_pb_deltaf", "cbar", ".", "upper_bump_profile", "filtered_upper_bump_profile"],
              ["lower_pb_deltaf", "cbar", ".", "lower_bump_profile", "filtered_lower_bump_profile"],
              ["ext_head_direction", "head_direction_offset", "head_direction_offset", "mean_bump_profile", "filtered_mean_bump_profile"]]
    fig, axd = plt.subplot_mosaic(mosaic=mosaic,
                                  figsize=(12, 5),
                                  width_ratios=[3,0.15,0.2,1,1],
                                  layout="constrained")

    # Plot Glutamate signal and external head direction
    # Both upper and lower arms
    vmin = np.min(expt.synced_df.iloc[:, :expt.num_rois].to_numpy())
    vmax = np.max(expt.synced_df.iloc[:, :expt.num_rois].to_numpy())
    plot_deltaf_heatmap(expt, axd["upper_pb_deltaf"], mode="upper", vmin=vmin, vmax=vmax, with_int_hd=True)
    plot_deltaf_heatmap(expt, axd["lower_pb_deltaf"], mode="lower", vmin=vmin, vmax=vmax, with_int_hd=True)
    plot_external_head_direction(expt, axd["ext_head_direction"])

    # Create colorbar
    norm = Normalize(vmin=vmin, vmax=vmax)
    fig.colorbar(ScalarMappable(norm=Normalize(vmin=vmin, vmax=vmax)),
                 cax=axd["cbar"], orientation='vertical', label='df/f')
    axd["cbar"].yaxis.set_label_position("left")

    # Plot head direction offset
    plot_hd_offset_histogram(expt, axd["head_direction_offset"])

    # Plot mean profile with cosine fit
    circ_r_min = 0.2
    # Upper arm
    upper_deltaf = np.array(expt.synced_df.iloc[:, expt.rois_per_arm:expt.rois_per_arm*2])
    upper_bumps = hd.align_bumps(upper_deltaf, offset=3)
    circ_rs = hd.get_circ_rs(upper_bumps)
    # filtered_upper_bumps = upper_bumps[circ_rs > circ_r_min]
    normalized_bumps = hd.normalize_bumps(upper_bumps)
    plot_bump_profile(upper_bumps, axd["upper_bump_profile"], with_std=True, with_fit=True)
    plot_bump_profile(normalized_bumps, axd["filtered_upper_bump_profile"], with_std=True, with_fit=True)

    # Lower arm
    lower_deltaf = np.array(expt.synced_df.iloc[:, 0:expt.rois_per_arm])
    lower_bumps = hd.align_bumps(lower_deltaf, offset=3)
    circ_rs = hd.get_circ_rs(lower_bumps)
    # filtered_lower_bumps = lower_bumps[circ_rs > circ_r_min]
    normalized_bumps = hd.normalize_bumps(lower_bumps)
    plot_bump_profile(lower_bumps, axd["lower_bump_profile"], with_std=True, with_fit=True)
    plot_bump_profile(normalized_bumps, axd["filtered_lower_bump_profile"], with_std=True, with_fit=True)

    # Average of both arms
    mean_deltaf = np.mean(np.array([upper_deltaf, lower_deltaf]), axis=0)
    mean_bumps = hd.align_bumps(mean_deltaf, offset=3)
    circ_rs = hd.get_circ_rs(mean_bumps)
    # filtered_mean_bumps = mean_bumps[circ_rs > circ_r_min]
    normalized_bumps = hd.normalize_bumps(mean_bumps)
    plot_bump_profile(mean_bumps, axd["mean_bump_profile"], with_std=True, with_fit=True)
    plot_bump_profile(normalized_bumps, axd["filtered_mean_bump_profile"], with_std=True, with_fit=True)

    # Plot aesthetics

    # Share y axes for bump profiles
    axd["filtered_upper_bump_profile"].sharey(axd["upper_bump_profile"])
    axd["filtered_lower_bump_profile"].sharey(axd["lower_bump_profile"])
    axd["filtered_mean_bump_profile"].sharey(axd["mean_bump_profile"])

    # Remove duplicate y ticklabels and labels
    for title in ["filtered_upper_bump_profile",
                  "filtered_lower_bump_profile",
                  "filtered_mean_bump_profile"]:
        # https://stackoverflow.com/questions/4209467/matplotlib-share-x-axis-but-dont-show-x-axis-tick-labels-for-both-just-one
        plt.setp(axd[title].get_yticklabels(), visible=False)
        axd[title].set_ylabel("")

    # Remove duplicate x ticklabels
    for title in ["upper_pb_deltaf", "upper_bump_profile", "filtered_upper_bump_profile",
                  "lower_pb_deltaf", "lower_bump_profile", "filtered_lower_bump_profile"]:
        axd[title].set_xticklabels([])
        axd[title].set_xlabel("")
    xticks = np.arange(0, axd['ext_head_direction'].get_xlim()[1], 300)
    xticklabels = [f"{tick/60:.0f}" for tick in xticks]
    axd['ext_head_direction'].set_xticks(xticks, xticklabels)

    # Set labels
    for title in ["upper_pb_deltaf", "lower_pb_deltaf"]:
        axd[title].set_ylabel("rois")
    axd["ext_head_direction"].set_ylabel("head direction\n(degrees)")
    axd["ext_head_direction"].set_xlabel("Time (minutes)")

    # deltaf titles
    for bump_region in ["upper", "lower"]:
        axd[f"{bump_region}_pb_deltaf"].set_title(f"{bump_region} arm")

    # Column titles
    axd["upper_bump_profile"].set_title("bump profile")
    # axd["filtered_upper_bump_profile"].set_title(f"filtered bump profile\nwith circular r > {circ_r_min}")
    axd["filtered_upper_bump_profile"].set_title(f"bumps individually normalized")

    # Row titles
    for bump_region in ["upper", "lower", "mean"]:
        axd[f"filtered_{bump_region}_bump_profile"].set_ylabel(f"{bump_region} bump")
        axd[f"filtered_{bump_region}_bump_profile"].yaxis.set_label_position("right")

    return fig, axd


##
# Behavior Plotting
##

def plot_path(ax, uvr, cax=None, downsample_factor=5):
    # Modified from unityvr.viz.viz.plotFlyPath

    # TODO: Downsample head direction scatterplot.
    # Maybe add argument with downsample factor?

    # plot path on given axes
    convfac = 10 # dc2cm

    # Plot path
    ax.plot(uvr.posDf.x*convfac,uvr.posDf.y*convfac,color='grey', linewidth=0.5)
    # Plot head direction at each point


    # Decimate points in path plot
    x = uvr.posDf.x[::downsample_factor] * convfac
    y = uvr.posDf.y[::downsample_factor] * convfac
    c = uvr.posDf.angle[::downsample_factor]
    # Plot path
    cb = ax.scatter(x, y, s=5, c=c, cmap='hsv')

    # Draw start point
    ax.plot(uvr.posDf.x[0]*convfac,uvr.posDf.y[0]*convfac,'ok')
    # Draw start text
    ax.text(uvr.posDf.x[0]*convfac+0.2,uvr.posDf.y[0]*convfac+0.2,'start')
    # Draw end point
    ax.plot(uvr.posDf.x.values[-1]*convfac,uvr.posDf.y.values[-2]*convfac,'sk')

    plt.colorbar(cb, cax=cax, ax=ax, label='head direction [degree]', pad=0.01, location='top')

    # Pick point close to bottom left of panel to place scale bar
    scale_bar_pos = [None]*2
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    scale_bar_pos[0] = (xlim[1] - xlim[0])* 0.1 + xlim[0]
    scale_bar_pos[1] = (ylim[1] - ylim[0])* 0.1 + ylim[0]
    uvrvisutils.plotScaleBar(ax,xlen=1,pos=scale_bar_pos,labeltext='1 cm')

    # Create legend for start and end point
    # start_point = mlines.Line2D([], [], color='black', marker='o', linestyle='None',
    #                       markersize=8, label='Start')
    # end_point = mlines.Line2D([], [], color='black', marker='s', linestyle='None',
    #                       markersize=8, label='End')
    # ax.legend(handles=[start_point, end_point])

    ax.set_xlabel("Fly Path")
    ax.set_aspect('equal')

def plot_dark_stim(uvr, fig=None, figsize=None):
    # Plot rotational and forward velocity histograms and fly path
    # Check if uvr object has computed velocities, if not compute them
    if not {'vT', 'vR'}.issubset(uvr.posDf.columns):
        posAnalysis.computeVelocities(uvr.posDf)

    # Create mosaic for plot layout
    mosaic = [['path', 'hist_forw'],
              ['path', 'hist_rot']]
    gridspec_kw = {"hspace": 0.3, "wspace": 0.35}
    if fig is None:
        fig, axd = plt.subplot_mosaic(mosaic, figsize=figsize,
                                      gridspec_kw=gridspec_kw)
        return_fig = True
    else:
        axd = fig.subplot_mosaic(mosaic,
                                 gridspec_kw=gridspec_kw)
        return_fig = False

    # Plot fly path
    plot_path(axd['path'], uvr)

    # Plot forward velocity
    axd['hist_forw'].hist(uvr.posDf.vT, bins=21, range=(0,6))
    axd['hist_forw'].set_xlabel("Forward Velocity (cm/s)")
    axd['hist_forw'].set_ylabel("Counts")

    # Plot rotational velocity
    axd['hist_rot'].hist(uvr.posDf.vR, bins=21, range=(-3,3))
    axd['hist_rot'].set_xlabel("Rotational Velocity (°/s)")
    axd['hist_rot'].set_ylabel("Counts")

    fig.suptitle("Dark Condition")

    if return_fig:
        return fig, axd
    else:
        return axd

def plot_grating_stim(uvr, fig=None, figsize=None):
    # Plot rotational velocity histograms for each stimuli speed and plot fly path

    # Check if uvr object has computed velocities, if not compute them
    if not {'vT', 'vR'}.issubset(uvr.posDf.columns):
        posAnalysis.computeVelocities(uvr.posDf)
    # Check if uvr object has derived head angle in rads, if not compute them
    if 'radangle' not in uvr.posDf.columns:
        posAnalysis.position(uvr, derive=True)
    # Check if unwrapped angle included, if not compute it
    if 'unwrapped_radangle' not in uvr.posDf.columns:
        uvr.posDf['unwrapped_radangle'] = np.unwrap(uvr.posDf.radangle)

    # Create mosaic for plot layout
    mosaic = [['path', 'angle_trace', 'angle_trace'],
              ['path', 'hist_rot', 'hist_forw']]
    gridspec_kw = {"hspace": 0.3, "wspace": 0.55,
                   "width_ratios": [2,1,1]}
    if fig is None:
        fig, axd = plt.subplot_mosaic(mosaic, figsize=figsize,
                                      gridspec_kw=gridspec_kw)
        return_fig = True
    else:
        axd = fig.subplot_mosaic(mosaic,
                                 gridspec_kw=gridspec_kw)
        return_fig = False

    # Plot fly path
    plot_path(axd['path'], uvr)

    # Plot angle over time
    axd['angle_trace'].plot(uvr.posDf.time, uvr.posDf.unwrapped_radangle)
    axd['angle_trace'].set_xticks(np.arange(0, uvr.posDf.time.iloc[-1]+1, 30))
    axd['angle_trace'].set_xlabel("Time (s)")
    axd['angle_trace'].set_ylabel("Head Angle (rad)")

    # Plot rotational velocity histogram
    axd['hist_rot'].hist(uvr.posDf.vR, bins=21, range=(-3,3))
    axd['hist_rot'].set_xlabel("Rotational Velocity (°/s)")
    axd['hist_rot'].set_ylabel("Counts")

    # Plot forward velocity histogram
    axd['hist_forw'].hist(uvr.posDf.vT, bins=21, range=(0,6))
    axd['hist_forw'].set_xlabel("Forward Velocity (cm/s)")
    axd['hist_forw'].set_yticks([])

    # Share y axis
    ylim_forw = axd['hist_forw'].get_ylim()
    ylim_rot = axd['hist_rot'].get_ylim()
    ylim = [min(ylim_forw[0], ylim_rot[0]), max(ylim_forw[1], ylim_rot[1])]

    axd['hist_forw'].set_ylim(ylim)
    axd['hist_rot'].set_ylim(ylim)

    fig.suptitle("Moving Grating Condition")

    if return_fig:
        return fig, axd
    else:
        return axd

def plot_closed_loop_stim(uvr, fig=None, figsize=None):
    # Plot angular head direction histogram and fly path

    # Check if uvr object has derived head angle in rads, if not compute them
    if 'radangle' not in uvr.posDf.columns:
        posAnalysis.position(uvr, derive=True)
    # Check if uvr object has computed velocities, if not compute them
    if not {'vT', 'vR'}.issubset(uvr.posDf.columns):
        posAnalysis.computeVelocities(uvr.posDf)

    # Create subplots
    if fig is None:
        fig = plt.figure(figsize=figsize)
        return_fig = True
    else:
        return_fig = False
    axd = {}
    axd['path'] = fig.add_subplot(1,2,1)
    axd['hist_angle'] = fig.add_subplot(1,2,2,
                                        projection='polar',
                                        theta_offset=np.pi/2)

    # Plot fly path
    plot_path(axd['path'], uvr)

    # Plot histogram of head direction when the fly is moving (filter out low velocity)
    radangle_filt = uvr.posDf[uvr.posDf['vT_filt'] > uvr.posDf['vT_filt'].quantile(0.25)]['radangle']

    counts, bins = np.histogram(radangle_filt, bins = int(360/15))
    axd['hist_angle'].bar(bins[:-1], counts, align='edge', width=np.diff(bins))

    axd['hist_angle'].set_xlabel("Head direction during movement")

    ## Set origin of hist so it looks like a donut
    axd['hist_angle'].set_rorigin(-1*max(counts))

    ## Set tick marks
    axd['hist_angle'].set_xticks(np.pi/180. * np.linspace(180,  -180, 8, endpoint=False))
    axd['hist_angle'].set_thetalim(-np.pi, np.pi)

    yticks = range(0, round(max(counts), -3), 1000)
    axd['hist_angle'].set_yticks(yticks)

    fig.suptitle("Closed Loop Condition")

    if return_fig:
        return fig, axd
    else:
        return axd

def plot_general_stim(uvr, fig=None, figsize=None):
    # Plot rotational velocity histograms for each stimuli speed and plot fly path

    # Check if uvr object has computed velocities, if not compute them
    if not {'vT', 'vR'}.issubset(uvr.posDf.columns):
        uvr.posDf = posAnalysis.computeVelocities(uvr.posDf)
    # Check if uvr object has derived head angle in rads, if not compute them
    if 'radangle' not in uvr.posDf.columns:
        uvr.posDf = posAnalysis.position(uvr, derive=True)
    # Check if unwrapped angle included, if not compute it
    if 'unwrapped_angle' not in uvr.posDf.columns:
        uvr.posDf['unwrapped_angle'] = np.unwrap(uvr.posDf.angle, period=360)

    # Create mosaic for plot layout
    mosaic = [['path', 'angle_trace', 'angle_trace'],
              ['path', 'hist_rot', 'hist_forw'],
              ['path', 'hist_angle', 'hist_angle']]
    gridspec_kw = {"hspace": 0.5, "wspace": 0.55,
                   "width_ratios": [4,1,1],
                   "height_ratios": [1,1,2]}
    per_subplot_kw={"hist_angle": {"projection": "polar",
                                   "theta_offset": np.pi/2}}
    if fig is None:
        fig, axd = plt.subplot_mosaic(mosaic, figsize=figsize,
                                      gridspec_kw=gridspec_kw,
                                      per_subplot_kw=per_subplot_kw)
        return_fig = True
    else:
        axd = fig.subplot_mosaic(mosaic,
                                 gridspec_kw=gridspec_kw,
                                 per_subplot_kw=per_subplot_kw)
        return_fig = False

    # Plot fly path
    plot_path(axd['path'], uvr)

    # Plot angle over time
    axd['angle_trace'].plot(uvr.posDf.time, uvr.posDf.unwrapped_angle)
    axd['angle_trace'].set_xticks(np.arange(0, uvr.posDf.time.iloc[-1]+1, 60))
    ymin = uvr.posDf.unwrapped_angle.min()
    ymax = uvr.posDf.unwrapped_angle.max()

    tick_size = np.floor((ymax - ymin)/5/360)*360
    if tick_size < 360:
        tick_size = 360
    yticks = np.arange(np.ceil(ymin/tick_size)*tick_size,
                       np.floor(ymax/tick_size)*tick_size,
                       tick_size)
    axd['angle_trace'].set_yticks(yticks)
    axd['angle_trace'].set_xlabel("Time (s)")
    axd['angle_trace'].set_ylabel("Head Angle (deg)")

    velocity_margin_cutoff = 0.03

    # Plot rotational velocity histogram
    mask = ((uvr.posDf['vR_filt'] > uvr.posDf['vR_filt'].quantile(velocity_margin_cutoff)) 
            & (uvr.posDf['vR_filt'] < uvr.posDf['vR_filt'].quantile(1 - velocity_margin_cutoff)))
    rot_vel_filt = uvr.posDf[mask]['vR_filt']
    axd['hist_rot'].hist(rot_vel_filt, bins=21)
    axd['hist_rot'].set_xlabel("Rotational Velocity (°/s)")
    axd['hist_rot'].set_ylabel("Counts")

    # Plot forward velocity histogram
    mask = ((uvr.posDf['vT_filt'] > uvr.posDf['vT_filt'].quantile(velocity_margin_cutoff)) 
            & (uvr.posDf['vT_filt'] < uvr.posDf['vT_filt'].quantile(1 - velocity_margin_cutoff)))
    forw_vel_filt = uvr.posDf[mask]['vT_filt']
    axd['hist_forw'].hist(forw_vel_filt, bins=21)
    axd['hist_forw'].set_xlabel("Forward Velocity (cm/s)")
    axd['hist_forw'].set_yticks([])

    # Plot histogram of head direction when the fly is moving (filter out low velocity)
    radangle_filt = uvr.posDf[uvr.posDf['vT_filt'] > uvr.posDf['vT_filt'].quantile(0.25)]['radangle']

    counts, bins = np.histogram(radangle_filt, bins = int(360/15))
    axd['hist_angle'].bar(bins[:-1], counts, align='edge', width=np.diff(bins))

    axd['hist_angle'].set_xlabel("Head direction during movement")

    ## Set origin of hist so it looks like a donut
    axd['hist_angle'].set_rorigin(-1*max(counts))

    ## Set tick marks
    axd['hist_angle'].set_xticks(np.pi/180. * np.linspace(180,  -180, 8, endpoint=False))
    axd['hist_angle'].set_thetalim(-np.pi, np.pi)

    yticks = range(0, round(max(counts), -3), 1000)
    axd['hist_angle'].set_yticks(yticks)

    # Share y axis
    ylim_forw = axd['hist_forw'].get_ylim()
    ylim_rot = axd['hist_rot'].get_ylim()
    ylim = [min(ylim_forw[0], ylim_rot[0]), max(ylim_forw[1], ylim_rot[1])]

    axd['hist_forw'].set_ylim(ylim)
    axd['hist_rot'].set_ylim(ylim)

    # fig.suptitle("Moving Grating Condition")

    if return_fig:
        return fig, axd
    else:
        return axd

##
# Imaging Plotting
##

def incr_bbox(bounding_box, image_shape, scale_factor):
    """Scale a bounding box keeping it centered at the same spot

    Args:
        bounding_box (ndarray): Bounding box of shape (2,2): [x or y, min or max]
        scale_factor (float): Amount to scale each side of the bounding box by

    Returns:
        NDArray[float64]: scaled bounding box
    """
    view_box = np.empty(shape=(2, 2))
    for dim in range(2):
        for lim in range(2):
            if lim == 0:  # min
                sign = -1
            if lim == 1:  # max
                sign = 1
            length = bounding_box[dim, 1] - bounding_box[dim, 0]
            scale_amount = sign * (scale_factor - 1) / 2 * length
            view_box[dim, lim] = bounding_box[dim, lim] + scale_amount
        # Clip box if it extends beyond image bounds
        if view_box[dim, 0] < 0:
            view_box[dim, 0] = 0
        if view_box[dim, 1] > image_shape[dim]:
            view_box[dim, 1] = image_shape[dim]
    return view_box

def get_bbox(rois, image_shape, scale_factor=1.5):
    """Given a list of rois, return a bounding box, a scale factor of 1 is a tight box

    Args:
        rois (list[ndarray]): List of rois, each roi is an ndarray of the points that make up the roi.
        scale_factor (float, optional): Amount to scale each side of the bounding box by. Defaults to 1.5.

    Returns:
        NDArray[float64]: Bounding box of shape (2,2): [x or y, min or max]
    """
    XCOL = 0
    YCOL = 1
    # roi_bound axes: [roi, x or y, min or max]
    roi_bounds = np.empty(shape=(len(rois), 2, 2))

    # Get min and max for each roi x and y
    for i, r in enumerate(rois):
        roi_bounds[i][0][0], roi_bounds[i][1][0] = r.min(axis=0)[XCOL : YCOL + 1]
        roi_bounds[i][0][1], roi_bounds[i][1][1] = r.max(axis=0)[XCOL : YCOL + 1]

    # Get the coords for the bounding box, using upper left corner to lower right
    # bounding_box axes: [x or y, min or max]
    bounding_box = np.empty(shape=(2, 2))
    bounding_box[:, 0] = roi_bounds[:, :, 0].min(axis=0)
    bounding_box[:, 1] = roi_bounds[:, :, 1].max(axis=0)

    # Create a larger bounding box to not cut off parts of the PB
    view_box = incr_bbox(bounding_box, image_shape, scale_factor)
    return view_box

def plot_roi(trial, panel):
    # Get pixel dimensions
    pixel_width = trial.tiff_metadata['pixel_width']
    pixel_width_unit = trial.tiff_metadata['width_unit']
    pixel_height = trial.tiff_metadata['pixel_height']
    pixel_height_unit = trial.tiff_metadata['height_unit']


    view_box = get_bbox(trial.rois, trial.mip_frame.shape, scale_factor=1.5)
    panel.imshow(trial.mip_frame)
    panel.axis('off')

    # Draw ROI's
    for j, r in enumerate(trial.rois):
        X_IDX = 0
        Y_IDX = 1
        panel.add_patch(
            Polygon(
                [[pt[Y_IDX], pt[X_IDX]] for pt in r],
                closed=True,
                fill=False,
                edgecolor=(1, 1, 1, 0.5),
                linewidth=0.5
            )
        )
        panel.text(
            r[:, Y_IDX].mean(),
            r[:, X_IDX].mean(),
            str(j + 1),
            dict(ha='center', va='center', fontsize=3.5, color='w'),
        )

    # Add scale bar
    x0, x1 = view_box[1]
    y0, y1 = view_box[0]

    # Get reasonable scalebar length
    scale_bar_length = round((x1-x0)*0.2, -1)
    if pixel_width_unit == "um":
        scale_bar_unit = r"$\mu m$"
    else:
        scale_bar_unit = pixel_width_unit

    scale_bar_margin = (x1-x0)*0.1
    scale_bar_x = x1 - scale_bar_margin
    scale_bar_y = y1 - scale_bar_margin
    scale_bar_width = scale_bar_length / float(pixel_width)
    scale_bar_height = (y1 - y0) * 0.0005
    scale_bar = Rectangle([scale_bar_x, scale_bar_y],
                          -1 * scale_bar_width,
                          -1 * scale_bar_height,
                          color='white',
                          fill=True)
    panel.add_patch(scale_bar)

    labelpad = (y1 - y0) * 0.001
    scale_bar_text_x = scale_bar_x - scale_bar_width/2
    scale_bar_text_y = scale_bar_y - scale_bar_height - labelpad
    panel.text(scale_bar_text_x, scale_bar_text_y,
               f"{scale_bar_length} {scale_bar_unit}",
               color='white',
               fontsize=5,
               ha='center',
               va='bottom')

    panel.set_xlim(view_box[1])
    panel.set_ylim(np.flip(view_box[0]))
    panel.set_title("Protocerebral Bridge\n glomeruli (ROI's)")
