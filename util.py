import numpy as np 
import pandas as pd 
import os 
import torch 
from skimage.io import imread, imsave
from skimage.morphology import remove_small_objects
from skimage.measure import *
import cv2

def draw_output(path, p_ellipse, i_ellipse):
    """
    Draws Ellipse on the image
    p_ellipse : Pupil ellipse
    i_ellipse : iris_ellipse
    """
    img = imread(path)
    cv2.ellipse(img, p_ellipse, (0,0,255), 2)
    cv2.ellipse(img, i_ellipse, (0,0,255), 2)
    return img 

def draw_ellipse(image, center, radius, angle=0, startAngle=0, 
                 endAngle=360, color=(0,0,255), thickness=5):
    center = center.round(0).astype('int')
    radius = radius.round(0).astype('int')
    image = cv2.ellipse(image, center, radius, 
                        angle, startAngle, endAngle, 
                        color, thickness)

    return image

def save_csv(score, name):
    """
    Save prediction scores as csv file
    """
    col = ['name', 'gt_pupil', 'pred_pupil', 'gt_iris', 'pred_pupil']
    df = pd.DataFrame(score, columns=col)
    df.to_csv(name, index=False)

      
def saveimage(pupil_boundary, iris_boundary, name, folder):
    """
    Inorder to visualize the prediction we can use this saveimage function
    """
    name = name[:-4]
    pupil_boundary = torch.sigmoid(pupil_boundary).cpu().squeeze(0).squeeze(0).detach().numpy()
    iris_boundary = torch.sigmoid(iris_boundary).cpu().squeeze(0).squeeze(0).detach().numpy()
    # debug 
    data = pd.read_csv('/content/testing_set/groundtruth/{}.csv'.format(name)).values
    data = np.delete(data, [4,-1])
    data = data.round(2)

    _, inner_prediction = cv2.threshold(pupil_boundary, 0.5, 1, 0)
    _, outer_prediction = cv2.threshold(iris_boundary, 0.5, 1, 0)
    inner_ellipse = fit_Ellipse((inner_prediction*127).astype('uint8'))
    outer_ellipse = fit_Ellipse((outer_prediction*127).astype('uint8'))                    
    # imsave(folder+name+'.png', pupil+iris)
    info = {
        'pupil_segmentation' : pupil_boundary,
        'iris_segmentation' : iris_boundary, 
        'p_ellipse': inner_ellipse,
        'i_ellipse': outer_ellipse,
    }
    return data, tuple2array(inner_ellipse), tuple2array(outer_ellipse), info  

def toNumpy(arr):
    return arr.cpu().squeeze(0).detach().numpy()

def tuple2array(ellipse):
    """
    Converts Ellipse tuple to array format center_x, center_y, radius_x, radius_y
    """
    factor = 0.5 # conversion factor from diameter to radius
    pred = np.array([ellipse[0][0], ellipse[0][1], ellipse[1][0]*factor, ellipse[1][1]*factor])
    return pred 

def ComputerMetric(file_name, gtmask, pred_mask, ellipse, mask_type):
    
    # load the ground truth 
    data = pd.read_csv(file_name).values
    data = np.delete(data, [4,-1])

    # computerIoU
    gtmask = gtmask.astype('uint8')
    pred_mask = pred_mask.astype('uint8')
    IoU = (pred_mask & gtmask).sum() / (pred_mask | gtmask).sum()
    
    pred = ellipse 

    if mask_type == 'pupil':
        coordinates = data[:4]
    else:
        coordinates = data[4:]
    
    # calculate the error
    error = abs(coordinates - pred) / coordinates
    return IoU.round(2), error.round(2)

def create_mask(coordinates, size=(224, 224)):

        mask = np.zeros(size, dtype='uint8')
        center = coordinates[:2]
        axis_len = coordinates[2:4]
        mask = draw_ellipse(mask, center, 
                            axis_len, color=1,
                            thickness=-1    
                            )
        return mask 

def create_folder(path):
    if not os.path.exists(path):
        os.makedirs(path)
    

def save_with_bbox(pupil_boundary, iris_boundary,box, name, folder):
    pupil_boundary = torch.sigmoid(pupil_boundary).cpu().squeeze(0).squeeze(0).detach().numpy()
    iris_boundary = torch.sigmoid(iris_boundary).cpu().squeeze(0).squeeze(0).detach().numpy()
    _, inner_prediction = cv2.threshold(pupil_boundary, 0.5, 1, 0)
    _, outer_prediction = cv2.threshold(iris_boundary, 0.5, 1, 0)
    out = 127 * (inner_prediction + outer_prediction)
    out = out.astype('uint8')
    box = box.cpu().squeeze(0).detach().numpy() * 224
    box = box.astype('int')
    out = cv2.ellipse(out, box[:2], box[2:4], 
                        0, 0, 360, 50, 5) 
    out = cv2.ellipse(out, box[4:6], box[6:8], 
                        0, 0, 360, 50, 5)                        
    imsave(folder+name+'.png', out)

def bbox2ellipse(bbox):
    cx = round((bbox[0] + bbox[2])/2, 2)
    cy = round((bbox[1] + bbox[3])/2, 2)
    rx = round((bbox[2] - bbox[0])/2, 2)
    ry = round((bbox[3] - bbox[1])/2, 2)
    return [cx,cy,rx,ry]

def fit_Ellipse(mask):
    # Ellipse_mask = np.zeros(mask.shape)
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    if len(contours) > 0 and len(contours[0]) > 5:
        cnt = contours[0]
        ellipse = cv2.fitEllipse(cnt)
        # cv2.ellipse(Ellipse_mask, ellipse, 1, -1)
    else:
        ellipse = ((0,0), (0,0), 0)
    return ellipse