# trial.py

from pathlib import Path
import logging
import pickle
import re
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.colors import CenteredNorm
import numpy as np
import pandas as pd
import pingouin

# from gulp2p import preproc
# from gulp2p.preproc import utils, imaging
from gulp2p.config import TRIAL_PICKLE_DIR, BHV_DATA_RAW_DIR

logger = logging.getLogger(__name__)

class Trial():
    def __init__(self, path, bhv_paths, tiff_metadata, mip_frame, slices, mc_obj, rois, masks, raw_flor, synced_df):
        assert path is not None
        self.path = Path(path) # path (Path): path to tiff file of fly trial
        self.name = path.stem
        self.bhv_paths = bhv_paths
        self.tiff_metadata = tiff_metadata
        self.mip_frame = mip_frame
        self.slices = slices
        self.mc_obj = mc_obj
        self.rois = rois
        self.masks = masks
        self.raw_flor = raw_flor
        self.synced_df = synced_df



    def get_num_rois(self):
        return len(self.rois)

    def shift_elements(self, arr, num, fill_value=np.nan):
        arr = np.roll(arr,num)
        if num < 0:
            arr[num:] = fill_value
        elif num > 0:
            arr[:num] = fill_value
        return arr

    # Analysis Functions

    def correlate_with_lag(self, a, v, lags):
        corrs = []
        for lag in lags:
            # Lag second array
            if lag < 0:
                lagged_a = a[:lag]
                lagged_v = v[-lag:]
            elif lag == 0:
                lagged_a = a
                lagged_v = v
            else:
                lagged_a = a[lag:]
                lagged_v = v[:-lag]

            corr = np.corrcoef(lagged_a, lagged_v)[0,1]
            corrs.append(corr)
        return corrs

    def get_roi_angles(num_rois):
        """Given the number of rois in one PB arm return a list sequentially assigning each roi to an angle

        Args:
            num_rois (int): number of rois in one PB arm (or in full EB)

        Returns:
            NDArray: Returns a list where at index i it has the angle for ROI i. 
        """
        return np.linspace(0,(2*np.pi),num_rois, endpoint=False)

    def angle_roi_to_radians(roi_angle, low=0, high=8):
        return roi_angle/(high-low) * 2*np.pi

    def angle_radians_to_roi(rad_angle, num_roi):
        roi_angle = rad_angle/(2*np.pi) * num_roi
        return roi_angle

    def get_mean_roi(roi_weights, precise=False):
        num_rois = len(roi_weights)
        angles = get_roi_angles(num_rois)
        circ_mean_rad = pingouin.circ_mean(angles=angles, w=roi_weights)
        if circ_mean_rad < 0:
            circ_mean_rad += 2*np.pi
        roi_angle = angle_radians_to_roi(circ_mean_rad, num_rois)
        if not precise:
            roi_angle = round(roi_angle)
        return roi_angle

    def get_mean_rois(deltaf_df, precise=False):
        num_frames, num_rois = deltaf_df.shape
        angles = np.repeat(np.expand_dims(get_roi_angles(num_rois), axis=0), num_frames, axis=0)
        circ_mean_rad = pingouin.circ_mean(angles=angles, w=deltaf_df, axis=1)
        # convert range -pi to pi into 0 to 2pi
        mask = (circ_mean_rad<0)
        circ_mean_rad[mask] += 2*np.pi

        roi_angle = angle_radians_to_roi(circ_mean_rad, num_rois)
        if not precise:
            roi_angle = np.round(roi_angle).astype(np.int32)
            roi_angle = roi_angle % num_rois
        return roi_angle

    def get_circ_r(roi_weights):
        num_rois = len(roi_weights)
        angles = get_roi_angles(num_rois)
        return pingouin.circ_r(angles=angles, w=roi_weights)

    def get_circ_rs(deltaf_df):
        num_frames, num_rois = deltaf_df.shape
        angles = np.repeat(np.expand_dims(get_roi_angles(num_rois), axis=0), num_frames, axis=0)
        return pingouin.circ_r(angles=angles, w=deltaf_df, axis=1)

    def filter_mean_rois(mean_rois, circ_rs, min_r=0.15):
        # in place change
        if np.issubdtype(mean_rois.dtype , np.integer):
            empty_value = -1
        else:
            empty_value = np.nan
        
        mask = (circ_rs <  min_r)
        mean_rois[mask] = empty_value
        return mean_rois
    
    def align_bumps(self, deltaf_df, mean_rois=None, min_r = 0, offset=0, shift_peak_flor=False, use_max_as_peak=False):
        aligned_bumps = deltaf_df.copy()

        if mean_rois is None:
            mean_rois = get_mean_rois(aligned_bumps, precise=False)

        if min_r > 0:
            circ_rs = get_circ_rs(aligned_bumps)
            filter_mean_rois(mean_rois, circ_rs)

        for frame, mean_roi in enumerate(mean_rois):

            roll_amount = -1 * mean_roi + offset
            aligned_bump = np.roll(aligned_bumps[frame, :], roll_amount)
            if shift_peak_flor:
                if use_max_as_peak:
                    peak_deltaf = np.max(aligned_bump)
                else:
                    peak_deltaf = aligned_bump[offset]
                aligned_bump += 1 - peak_deltaf
            aligned_bumps[frame, :] = aligned_bump

            # if frame < 10:
            #     print(mean_roi)
            #     print(peak_deltaf)
            #     print(aligned_bump)

        return aligned_bumps

    # Plotting Functions

    def plot_roi(self, trial_dat, panel, title="Protocerebral Bridge\n glomeruli (ROI's)", full_img=False):
        # ['trialName', 'fpv', 'meanMIP_G', 'ROIOutlines', 'allrois', 'rawF_G', 'DF_G']
        # Get pixel dimensions
        trial_nm = trial_dat.get('trial_nm', trial_dat.get('trialName'))
        # pixel_dims = trial_dat.get('pixel_dims' ,iPP.getPixelDims(trial_nm))
        rois = trial_dat.get('rois', trial_dat.get('allrois'))
        mean_stack = trial_dat.get('stack_mean', trial_dat.get('meanMIP_G'))

        view_box = iPP.get_bbox(rois, image_shape=mean_stack.shape, scale_factor=1.5)
        panel.imshow(mean_stack)
        panel.axis('off')

        # Draw ROI's
        for j, r in enumerate(rois):
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

        # # Add scale bar
        # x0, x1 = view_box[1]
        # y0, y1 = view_box[0]

        # # Get reasonable scalebar length
        # scale_bar_length = round((x1-x0)*0.2, -1)
        # if pixel_dims['width_unit'] == "um":
        #     scale_bar_unit = r"$\mu m$"
        # else:
        #     scale_bar_unit = pixel_dims['width_unit']

        # scale_bar_margin = (x1-x0)*0.1
        # scale_bar_x = x1 - scale_bar_margin
        # scale_bar_y = y1 - scale_bar_margin
        # scale_bar_width = scale_bar_length / float(pixel_dims['pixel_width'])
        # scale_bar_height = (y1 - y0) * 0.0005
        # scale_bar = Rectangle([scale_bar_x, scale_bar_y],
        #                       -1 * scale_bar_width,
        #                       -1 * scale_bar_height,
        #                       color='white',
        #                       fill=True)
        # panel.add_patch(scale_bar)

        # labelpad = (y1 - y0) * 0.001
        # scale_bar_text_x = scale_bar_x - scale_bar_width/2
        # scale_bar_text_y = scale_bar_y - scale_bar_height - labelpad
        # panel.text(scale_bar_text_x, scale_bar_text_y,
        #            f"{scale_bar_length} {scale_bar_unit}",
        #            color='white',
        #            fontsize=5,
        #            ha='center',
        #            va='bottom')

        if not full_img:
            panel.set_xlim(view_box[1])
            panel.set_ylim(np.flip(view_box[0]))
        panel.set_title(title)

    def plot_heatmap(self, flor, axes=None, aspect=32, plot_cbar=True):
        if axes is None: 
            fig = plt.figure()
            ax = fig.add_subplot()
        else:
            ax = axes
            fig = axes.get_figure()

        im = ax.imshow(flor.T, aspect=aspect, interpolation='none', norm=CenteredNorm(), cmap='PiYG')
        ax.invert_yaxis()

        if plot_cbar:
            cbar = fig.colorbar(im, ax=ax, shrink=0.5)
            clim = [np.min(flor), np.max(flor)]
            cbar.ax.set_ylim(clim)
            cbar.set_ticks([clim[0], 0, clim[1]])

        if axes is not None:
            return
        else:
            return fig, ax

    def plot_tuning_curves(self, expDf, ax, bot_roi=0, top_roi=7, scatter=True, fit=True, color=None):
        for roi in range(bot_roi, top_roi+1):
            x = expDf['angle']
            y = expDf[f'roi{roi}']
            poly_coefs = np.polyfit(x, y, 4)
            p = np.poly1d(poly_coefs)

            if scatter:
                ax.scatter(expDf['angle'], expDf[f'roi{roi}'])
            if fit:
                ax.plot([p(point) for point in np.linspace(0, 360,360)], color= color)
        
        # return ax





def parse_trial_name(trial_path, cell_types, genetic_tools):
    # Return fields defined in trials name
    # ex: 20240215_DR018xiGluSnFR_Fly02_00005.tif
    # regex patterns built with https://regexr.com/

    # Define regex patterns
    patterns = {
        "date": r"^(\d{8})_",
        "line": r"([^_]+x[^_]+)",
        "cell_type": r"([^_]+)x[^_]+",
        "genetic_tool": r"[^_]+x([^_]+)",
        "fly": r"fly(\d+)",
        "trial": r"_(\d{5})\."
    }

    # Get info from trial name with regex
    metadata = {}
    for name, pattern in patterns.items():
        match = re.search(pattern, trial_path.name,
                          flags=re.IGNORECASE)
        if match is None:
            metadata[name] = None
        else:
            metadata[name] = match.group(1)

    # Assign default values
    # Convert from strings and set default values.
    if metadata['date'] is None:
        # Get date from path name if missing.
        metadata['date'] = trial_path.parent.name
    if metadata['cell_type'] is not None:
        # Correct case and zero-padding of cell_type
        cell_type = metadata['cell_type']
        match = re.search(r"\d", cell_type)
        if match is not None:
            prefix = cell_type[:match.start()]
            suffix = int(cell_type[match.start():])
            cell_type = f"{prefix}{suffix:03d}"
        try:
            find_idx = list(map(str.lower, cell_types)).index(cell_type.lower())
            metadata['cell_type'] = cell_types[find_idx]
        except:
            pass
    if metadata['genetic_tool'] is not None:
        # Correct case of genetic_tool
        try:
            find_idx = list(map(str.lower, genetic_tools)).index(metadata['genetic_tool'].lower())
            metadata['genetic_tool'] = genetic_tools[find_idx]
        except:
            pass
    if metadata['fly'] is not None:
        # Convert fly number to int.
        metadata['fly'] = int(metadata['fly'])
    else:
        # Set default value of 1.
        metadata['fly'] = 1
    if metadata['trial'] is not None:
        # Convert trial number to int.
        metadata['trial'] = int(metadata['trial'])

    return metadata

def create_trial_df(root_tiff_folder, cell_types, genetic_tools):
    trial_df_dicts = []
    for path in root_tiff_folder.rglob('*.tif'):
        trial_metadata = parse_trial_name(path, cell_types, genetic_tools)

        trial_df_dicts.append({
            "path": path,
            **trial_metadata
        })
    trial_df = pd.DataFrame(trial_df_dicts)
    return trial_df
