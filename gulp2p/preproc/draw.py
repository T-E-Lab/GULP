import numpy as np
import cv2 as cv
import pandas as pd

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

def PolyROIs(viewer, stack):
    """ Make polygonal ROIs from a napari layer
    """
    # Get the ROIs from napari
    rois = viewer.layers['Shapes'].data

    # Initialize an array to hold the ROI masks
    masks = []

    # Make the polygonal ROIs from the points
    for roi in rois:
        mask = np.zeros(stack.shape)
        masks.append(cv.fillPoly(mask,[np.array(np.flip(roi),dtype='int')],1).T)

    return [rois, rois, masks]

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
