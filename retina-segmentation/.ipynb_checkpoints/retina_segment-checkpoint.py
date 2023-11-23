import cv2
import numpy as np
import matplotlib.pyplot as plt
import math

 
def segmentation(img, t=8, A=200, L=50, resize=True):  
    """
    Employes a global thresholding based segmentation algorithm for retinal vessel segmentation

    Args:
        t: Threshold => the threshold used to segment the image (value of 8-10 works best. Otsu and Isodata values do not led to best result)
        A: Threshold area => All the segments less than A in area are to be removed and considered as noise
        L: Threshold length => All the centrelines less than L in length are to be removed
        resize: boolean => weather you want to resize the image to (1000px, 1000px)
    Returns:
        Segmented image
    """

    # Resize image to ~(1000px, 1000px) for best results
    if resize:
        img = cv2.resize(img, (1000, 1000))

    # Green Channel
    g = img[:,:,1]
    
    # Threshold the image
    # _, thresholded = cv2.threshold(img, 30, 255, cv2.THRESH_BINARY)
    #Creating mask for restricting FOV
    _, mask = cv2.threshold(g, 10, 255, cv2.THRESH_BINARY)  
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    mask = cv2.erode(mask, kernel, iterations=3)
    
    # CLAHE and background estimation
    clahe = cv2.createCLAHE(clipLimit = 3, tileGridSize=(9,9))
    g_cl = clahe.apply(g)
    g_cl1 = cv2.medianBlur(g_cl, 5)
    bg = cv2.GaussianBlur(g_cl1, (55, 55), 0)

    # Background subtraction
    norm = np.float32(bg) - np.float32(g_cl1)
    norm = norm*(norm>0)

    # Thresholding for segmentation
    _, t = cv2.threshold(norm, t, 255, cv2.THRESH_BINARY)

    # Removing noise points by coloring the contours
    t = np.uint8(t)
    th = t.copy()
    contours, hierarchy = cv2.findContours(t, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    for c in contours:
        if ( cv2.contourArea(c)< A):
            cv2.drawContours(th, [c], 0, 0, -1)
    th = th*(mask/255)
    th = np.uint8(th)
    #plt.imshow(th, cmap='gray')  # THE SEGMENTED IMAGE
        
    return th
