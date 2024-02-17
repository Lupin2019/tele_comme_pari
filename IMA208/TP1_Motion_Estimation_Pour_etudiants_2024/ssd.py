import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt
import cv2
from scipy.ndimage import gaussian_filter
from scipy.interpolate import griddata
from scipy.signal import convolve2d


from utils import *

def me_ssd(cur, ref, brow, bcol, search, lamb=0):

    ref_extended = cv2.copyMakeBorder(ref, search, search, search, search, cv2.BORDER_REPLICATE) # To avoid border effect
    prediction = np.zeros_like(cur);
    mvf = np.zeros((*cur.shape,2))
    

    if lamb == 0:
        for r in range(0, cur.shape[0], brow):
            for c in range(0, cur.shape[1], bcol):
                # current block
                B = cur[r:r+brow, c:c+bcol]

                # Init
                costMin = np.inf
                Rbest = None
                Vbest = None

                for drow in range(-search, search + 1):
                    for dcol in range(-search, search + 1):
                        B_ref = ref_extended[
                            search+r+drow : search+r+drow+B.shape[0],
                            search+c+dcol : search+c+dcol+B.shape[1],
                        ]

                        cost = np.mean((B - B_ref) ** 2) # MSE 

                        if cost < costMin:
                            Rbest = B_ref
                            Vbest = (drow, dcol)
                            costMin = cost
                
                mvf[r : r+brow, c : c+bcol, 0] = np.ones((brow, bcol)) * Vbest[0] # Once the loop is over, save the best row displacement field
                mvf[r : r+brow, c : c+bcol, 1] = np.ones((brow, bcol)) * Vbest[1] # Once the loop is over, save the best column displacement field
                prediction[r:r+brow,c:c+bcol] = Rbest
    
    else:
        for r in range(0, cur.shape[0], brow):
            for c in range(0, cur.shape[1], bcol):
                # current block
                B = cur[r:r+brow, c:c+bcol]

                # Init
                costMin = np.inf
                Rbest = None
                Vbest = None

                # Neighbours : pV is the regularization vector. The regularizer must be such that the estimated displacement is not too far away from pV
                pV = computePredictor(r,c,brow,bcol,mvf,ref,cur)

                for drow in range(-search, search + 1):
                    for dcol in range(-search, search + 1):
                        B_ref = ref_extended[
                            search+r+drow : search+r+drow+B.shape[0],
                            search+c+dcol : search+c+dcol+B.shape[1],
                        ]

                        lamb_area = lamb * B.shape[0] * B.shape[1]

                        cost = np.mean((B - B_ref) ** 2) + lamb_area * np.sum((np.array([drow, dcol]) - pV) ** 2)

                        if cost < costMin:
                            Rbest = B_ref
                            Vbest = (drow, dcol)
                            costMin = cost
                
                mvf[r : r+brow, c : c+bcol, 0] = np.ones((brow, bcol)) * Vbest[0] # Once the loop is over, save the best row displacement field
                mvf[r : r+brow, c : c+bcol, 1] = np.ones((brow, bcol)) * Vbest[1] # Once the loop is over, save the best column displacement field
                prediction[r:r+brow,c:c+bcol] = Rbest

    return -mvf, prediction