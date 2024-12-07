import numpy as np
import pandas as pd
import xarray as xr
import cv2 as cv
import logging

logger = logging.getLogger(__name__)

def WedgeROIs(viewer, stackMean):
    """ Get an ellipse and divide it into 16 wedges
    """
    numROIs = 16
    angStep = 360/numROIs

    EBOutline = viewer.layers["Shapes"]

    ellipseCent = [int(np.mean([p[-2] for p in EBOutline.data[0]])),
              int(np.mean([p[-1] for p in EBOutline.data[0]]))]
    ellipseCentInt = (int(ellipseCent[0]),int(ellipseCent[1]))
    ellipseAx1 = np.sqrt((EBOutline.data[0][2][-1] - EBOutline.data[0][1][-1])**2 +
                        (EBOutline.data[0][2][0] - EBOutline.data[0][1][0])**2)
    ellipseAx2 = np.sqrt((EBOutline.data[0][0][-1] - EBOutline.data[0][1][-1])**2 +
                        (EBOutline.data[0][0][0] - EBOutline.data[0][1][0])**2)
    ellipseAng = 180/np.pi*np.arcsin((EBOutline.data[0][0][0] - EBOutline.data[0][1][0])/
                                    (EBOutline.data[0][0][-1] - EBOutline.data[0][1][-1]))

    rois = []
    allMasks = []
    for a in range(0,numROIs):
        mask = np.zeros((stackMean.shape[0], stackMean.shape[1]))
        pts = cv.ellipse2Poly(ellipseCentInt,
                              (int(0.5*ellipseAx1), int(0.5*ellipseAx2)),
                              int(ellipseAng),
                              int(angStep*(a-1)),int(angStep*a),
                              3)
        roiNow = np.append(pts, [np.array(ellipseCentInt)], axis=0)
        rois.append(roiNow)
        allMasks.append(cv.fillConvexPoly(mask,roiNow,1).T)

    return [EBOutline.data, rois, allMasks]

def PolyROIs(viewer, stack_sizes):
    """ ake polygonal ROIs from a napari layer

    Args:
        viewer (napari.Viewer): Napari viewer with 'Shapes' layer containing rois
        stack_sizes (dict): dictionary with dimension names and their sizes. (accessible with xr.DataArray.sizes) 

    Returns:
        _type_: _description_
    """
    # Get the ROIs from napari
    rois = viewer.layers['Shapes'].data

    # If there are multiple channels, rois are made up of 3d points with the channel index as the z value.
    # Reshape rois from shape: [num rois, num dims] -> shape: [num channels, num rois, num dims]
    reshaped_rois = []
    for ch_idx in range(stack_sizes['C']):
        channel_rois = []
        for roi in rois:
            if roi.shape[1] == 3:
                # If roi has 3 dimensional points, select only the points in the current channel and remove the z dimension.
                reshaped_roi = roi[roi[:, 0] == ch_idx, 1:]
            if roi.shape[1] == 2:
                # If roi has 2 dimensional points, append as normal. Should only have 1 channel.
                reshaped_roi = roi
            if reshaped_roi.size == 0:
                continue
            channel_rois.append(reshaped_roi)
        reshaped_rois.append(channel_rois)
    rois = reshaped_rois

    # Initialize an array to hold the ROI masks
    masks = []

    # Make the polygonal ROIs from the points
    for ch_idx in range(stack_sizes['C']):
        channel_masks = []
        for roi in rois[ch_idx]:
            ch_mask = xr.DataArray(data=np.zeros(shape=(stack_sizes['Y'], stack_sizes['X'])),
                                   dims=['Y', 'X'])
            ch_roi = np.array(roi, dtype='int')

            cv.fillPoly(img=ch_mask.to_numpy(), pts=[ch_roi], color=1)
            # opencv's image coord convention is different. Need to rotate mask 90 deg clockwise
            ch_mask.data = np.rot90(ch_mask.to_numpy())
            channel_masks.append(ch_mask)
        # masks.append(cv.fillPoly(mask,[np.array(np.flip(roi),dtype='int')],1).T)
        masks.append(channel_masks)

    return rois, masks

def EBROI(viewer, stack):
    """ Make a doughnut shaped ROI
    """

    EBOutline = viewer.layers["Shapes"]

    ellipseCent = [int(np.mean([p[-2] for p in EBOutline.data[0]])),
              int(np.mean([p[-1] for p in EBOutline.data[0]]))]
    ellipseCentInt = (int(ellipseCent[0]),int(ellipseCent[1]))
    ellipseAx1 = np.sqrt((EBOutline.data[0][2][-1] - EBOutline.data[0][1][-1])**2 +
                        (EBOutline.data[0][2][0] - EBOutline.data[0][1][0])**2)
    ellipseAx2 = np.sqrt((EBOutline.data[0][0][-1] - EBOutline.data[0][1][-1])**2 +
                        (EBOutline.data[0][0][0] - EBOutline.data[0][1][0])**2)
    ellipseAng = 180/np.pi*np.arcsin((EBOutline.data[0][0][0] - EBOutline.data[0][1][0])/
                                    (EBOutline.data[0][0][-1] - EBOutline.data[0][1][-1]))

    pts = cv.ellipse2Poly(ellipseCentInt,
                          (int(0.5*ellipseAx1), int(0.5*ellipseAx2)),
                          int(ellipseAng),
                          0, 360,
                          3)

    # Initialize an array to hold the ROI masks
    mask = np.zeros((stack.shape[0], stack.shape[1]))
    allMasks = []
    allMasks.append(cv.fillConvexPoly(mask,pts,1).T)

    return [EBOutline.data, pts, allMasks]
