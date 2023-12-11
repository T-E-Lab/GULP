# Move these functions to a new package associated with plotting, analysis in the future.
import numpy as np
import pandas as pd

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