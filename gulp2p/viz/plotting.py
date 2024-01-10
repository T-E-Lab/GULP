# Move these functions to a new package associated with plotting, analysis in the future.
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
# import matplotlib.lines as mlines
from skimage.measure import block_reduce 

# import unityvr.preproc.logproc as lp
from unityvr.analysis import posAnalysis
import unityvr.viz.utils as uvrvisutils

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
    else:
        axd = fig.subplot_mosaic(mosaic,
                                 gridspec_kw=gridspec_kw)

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

    if fig is None:
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
    else:
        axd = fig.subplot_mosaic(mosaic,
                                 gridspec_kw=gridspec_kw)
    
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

    if fig is None:
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

    if fig is None:
        return fig, axd
    else:
        return axd


