from torch.utils.data import Dataset, DataLoader
from albumentations import (ShiftScaleRotate, Compose, PadIfNeeded, HorizontalFlip, OneOf, ElasticTransform,
                             OpticalDistortion, Normalize, RandomGamma, LongestMaxSize, GaussNoise, Resize, VerticalFlip)
from albumentations.pytorch import ToTensorV2                             
import cv2
import os
from glob import glob
import torch
import matplotlib.pyplot as plt
from skimage.io import imread
import numpy as np
import pandas as pd 
from util import draw_ellipse

class CustomDatareader(Dataset):

    def __init__(self, 
                 image_folder:str,
                 gt_path:str, 
                 size:tuple=(224, 224), 
                 data_type:str = 'train',
                 ifbbox:bool = False,
                 val:int = 1,
                ):
        '''
        For this project we donot have a custom train and validation dataset, 
        Hence we use the variable val 
        val : value between [1, 10], if the image index is divisible by val, its added to validation set
        '''
                
        self.size = size
        self.data_type = data_type
        self.val = val
        self.img, self.gt = self.getImageList(image_folder, gt_path)
        self.length = len(self.img)
        self.ifbbox = ifbbox
        
        if data_type == 'train':
            self.transforms = Compose([Resize(height=size[0], width=size[1], interpolation=cv2.INTER_NEAREST),
                                       ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=20, p=0.7,
                                                        border_mode=cv2.BORDER_CONSTANT, value=0),
                                       HorizontalFlip(p=0.5),
                                       OneOf([ElasticTransform(p=1, alpha=50, sigma=30, alpha_affine=30,
                                                               border_mode=cv2.BORDER_CONSTANT, value=0),
                                              OpticalDistortion(p=1, distort_limit=0.5, shift_limit=0.1,
                                                                border_mode=cv2.BORDER_CONSTANT, value=0)], p=0.5),
                                       RandomGamma(gamma_limit=(80, 120), p=0.5),
                                       GaussNoise(var_limit=(0.02, 0.1), mean=0, p=0.5),
                                       Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                                       ToTensorV2(),
                                       ])

        elif data_type == 'val' or data_type == 'test':
            self.transforms = Compose([LongestMaxSize(max_size=max(size)),
                                       PadIfNeeded(min_height=size[0], min_width=size[1], value=0,
                                                   border_mode=cv2.BORDER_CONSTANT),
                                       Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                                       ToTensorV2(),            
                                       ])
        else:
            print('\n -------- Wrong data type ------ \n')
            # TODO make this as a warning 

    def __getitem__(self, idx):
        
        assert self.img[idx].split('/')[-1][:-4] == self.gt[idx].split('/')[-1][:-4]
        
        image = np.zeros((self.size[0], self.size[1], 3), dtype='uint8')
        image[:160, :224, :] = imread(self.img[idx])
        data = pd.read_csv(self.gt[idx]).values[0]
        
        # pupil mask
        pupil_mask = self._create_mask(data[:4])
        
        # iris mask
        iris_mask = self._create_mask(data[5:9])

        local = np.concatenate((pupil_mask[:, :, np.newaxis],
                                    iris_mask[:, :, np.newaxis]), axis=2)
        augmented = self.transforms(image=image, mask=local)
        image, local = augmented['image'], augmented['mask']
        pupil_mask = local[:, :, 0]
        iris_mask = local[:, :, 1]

        image = image.float()
        pupil_mask = pupil_mask.float().unsqueeze(0)
        iris_mask = iris_mask.float().unsqueeze(0)
        name = self.img[idx].split('/')[-1]
        
        cord = np.delete(data, [4,-1]) / self.size[0]
        if self.ifbbox:
            # if intrested in Bbox loss CIoU then we need to convert [cx,cy,rx,ry] to [xmin,ymin, xmax, ymax] 
            cord = np.array(self._convert2box(cord[:4]) + self._convert2box(cord[4:])).astype('float32') 
        else:
            cord = cord.astype('float32')    
        if self.data_type == 'val' or self.data_type == 'test':
            return image, pupil_mask, iris_mask, cord, name 
        return image, pupil_mask, iris_mask, cord

    def __len__(self):
        return self.length
    
    def _convert2box(self, ellipse):
        
        # Convert [center_x, center_y, axis_x, axis_y] -- [xmin, ymin, xmax, ymax]

        cx,cy = ellipse[0], ellipse[1]
        rx, ry = ellipse[2], ellipse[3]
        bbox = [cx-rx, cy-ry, cx+rx, cy+ry]
        return bbox 

    def _testmask(self, mask):
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        ax.imshow(mask)
        ax.show()

    def _create_mask(self, coordinates):
        # creates mask from the elliptical coordinates 
        mask = np.zeros(self.size, dtype='uint8')
        center = coordinates[:2]
        axis_len = coordinates[2:4]
        mask = draw_ellipse(mask, center, 
                            axis_len, color=1,
                            thickness=-1    
                            )
        return mask

    def getImageList(self, image_folder, gt_path):

        """
        In the case of train set, we split the validation data from the Train set. 
        Using the index given while initializing the function.

        Test does not have to be split.  

        """
        train_image = []
        train_gt = []
        val_image = []
        val_gt = []

        img = sorted(glob(image_folder+'*.png'))
        labels = sorted(glob(gt_path+'*.csv'))

        for enum, (image, label) in enumerate(zip(img, labels)):
            if enum % 10 == self.val:
                val_image.append(image)
                val_gt.append(label)
            else:
                train_image.append(image)
                train_gt.append(label)

        if self.data_type == 'train':
            return train_image, train_gt
        
        elif self.data_type == 'val':
            return val_image, val_gt

        else: 
            # Test set no need to split
            return img, labels 