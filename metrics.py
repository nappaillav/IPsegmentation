import numpy as np 
import pandas as pd 
import cv2 


def compute_metric(gt, pred):
    """
    gt : ground truth  
    pred : prediction 
    """
    py_error = compute_error(pred[3], gt[3])
    px_error = compute_error(pred[2], gt[2])

    pIOU = compute_IoU(gt[:4], pred[:4])
    return [px_error, py_error, pIOU]


def compute_error(pred, gt):
    return abs(pred - gt)/ gt

def draw_ellipse(image, center, radius, angle=0, startAngle=0, endAngle=360, color=(0,0,255), thickness=5):
    center = center.round(0).astype('int')
    radius = radius.round(0).astype('int')
    image = cv2.ellipse(image, center, radius, 
                           angle, startAngle, endAngle, color, thickness)

    return image


def compute_IoU(gt, pred):

    gt_image = np.zeros((224, 224), dtype='uint8')
    pred_image = np.zeros((224, 224), dtype='uint8')

    gt_image = cv2.ellipse(gt_image, (int(gt[0]), int(gt[1])), (int(gt[2]), int(gt[3])), 0, 0, 360, 1, -1)
    pred_image = cv2.ellipse(pred_image, (int(pred[0]), int(pred[1])), (int(pred[2]), int(pred[3])), 0, 0, 360, 1, -1)

    IoU = (pred_image & gt_image).sum() / (pred_image | gt_image).sum()
    return IoU

def convert2float(s):
    return np.float_(s.strip('][').split(', '))

if __name__ == '__main__':
    file_names = ['./model_1/Unet_DB_R34_reg.csv', './model_1/Unet_DB_R34_seg.csv', 
                 './model_3/Unet_DB_CIoU_R34_reg.csv', './model_3/Unet_DB_CIoU_R34_seg.csv']
    results = []
    for file_name in file_names:
        data = pd.read_csv(file_name)
        col = ['name', 'gt_pupil', 'pred_pupil', 'gt_iris', 'pred_pupil']
        

        data = data.values
        dataset = []
        pupil_scores = []
        iris_scores = []
        # results = []
        for ind in range(len(data)):
            name = data[ind][0]
            score = compute_metric(convert2float(data[ind][1]),  
                                  convert2float(data[ind][2]))
            
            pupil_scores.append(np.array(score))
            
            score = compute_metric(convert2float(data[ind][3]),  
                                  convert2float(data[ind][4]))
            iris_scores.append(np.array(score))

        pupil_scores = list((np.array(pupil_scores).mean(0)*100).round(2))
        iris_scores = list((np.array(iris_scores).mean(0)*100).round(2))

        temp = [file_name]
        temp.extend(pupil_scores + iris_scores)
        print(temp)
        results.append(temp)
    df = pd.DataFrame(results, columns=['name','error_px', 'error_py', 'pIoU', 'error_ix', 'error_iy', 'iIoU'])