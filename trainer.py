import numpy as np 
import cv2
import os  
import math
import torch.nn as nn 
from dataset import CustomDatareader
from loss_function import DiceLoss, DiceBCELoss, IoULoss, FocalLoss
from torch.utils.data import DataLoader
import torch.optim as optim
import torch
from model import UnetModel
from torch.optim.lr_scheduler import MultiStepLR
from skimage.io import imsave, imread
import pandas as pd 
from torchvision.ops import complete_box_iou_loss
import wandb
from util import * 

def train(cfg:dict):


    if cfg.IOU_loss == False:
        cfg.weight_ciou = 0

    create_folder(cfg.save_dir)

    # TODO load dataset
    train_data = CustomDatareader(cfg.train_folder,
                                cfg.train_gt, 
                                cfg.size, 
                                data_type = 'train'
                                )

    train_dataloader = DataLoader(dataset=train_data, batch_size=cfg.BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
    val_data = CustomDatareader(image_folder=cfg.val_folder,
                                gt_path=cfg.val_gt, 
                                size=cfg.size, 
                                data_type = 'val'
                                )
    val_dataloader = DataLoader(dataset=val_data, batch_size=cfg.BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

    seg_model = UnetModel().to(cfg.device) # TODO Default resnet34
    
    train_data_len = train_data.length 
    val_data_len = val_data.length 

    criterion = DiceBCELoss(weight=(0.5, 0.5))                                                                                                                                                  
    
    optimizer = optim.Adam(seg_model.parameters(), lr=cfg.lr_max, weight_decay=cfg.L2)

    scheduler = MultiStepLR(optimizer, milestones=[int((5 / 10) * cfg.epochs),
                                                int((8 / 10) * cfg.epochs)], gamma=0.1, last_epoch=-1)

    # Parameters 
    best_inner_dice = 0.0
    best_outer_dice = 0.0
    img = []
    local_inner = []
    local_outer = []
    mse = nn.MSELoss()
    l1 = nn.L1Loss()

    for epoch in range(cfg.epochs):

        seg_model.train()

        epoch_train_totalloss = []
        n_steps_per_epoch = math.ceil(train_data_len / cfg.BATCH_SIZE)

        for enum, (inp, pupil_boundary, iris_boundary, cord) in enumerate(train_dataloader):

            inp = inp.to(cfg.device)
            pupil_boundary = pupil_boundary.to(cfg.device)
            iris_boundary = iris_boundary.to(cfg.device)
            cord = cord.to(cfg.device)

            pred_pupil, pred_iris, bbox = seg_model(inp)
            
            # Pupil loss 
            pupil_loss = criterion(pred_pupil, pupil_boundary)
            pupil_bce, pupil_dice = criterion.bce_loss, criterion.dice_loss

            # Iris Loss
            iris_loss = criterion(pred_iris, iris_boundary)
            iris_bce, iris_dice = criterion.bce_loss, criterion.dice_loss
            if cfg.IoULoss:
                ciou_pupil = complete_box_iou_loss(bbox[:, :4], cord[:, :4], reduction='mean') 
                ciou_iris = complete_box_iou_loss(bbox[:, 4:], cord[:, 4:], reduction='mean')
                ciou_loss = cfg.weight_pupil_ciou * ciou_pupil + cfg.weight_iris_ciou * ciou_iris 
                bb_loss = cfg.weight_l1 * l1(bbox, cord) + cfg.weight_mse * mse(bbox, cord) + cfg.weight_ciou * ciou_loss
            else:    
                bb_loss = cfg.weight_l1 * l1(bbox, cord) + cfg.weight_mse* mse(bbox, cord)
            

            optimizer.zero_grad()
            total_loss = pupil_loss + iris_loss + bb_loss
            total_loss.backward()
            optimizer.step()

            metrics = {"train/pupil_bce": pupil_bce,
                       "train/pupil_dice": pupil_dice, 
                       "train/iris_bce": pupil_bce,
                       "train/iris_dice": pupil_dice, 
                       "train/total_loss": total_loss, 
                       "train/epoch": (enum + 1 + (n_steps_per_epoch * epoch)) / n_steps_per_epoch, 
                       }
            wandb.log(metrics)
            metric_bbox={
                        "train/bbox_reg_loss": bb_loss, 
                        }
            wandb.log(metric_bbox)
            if cfg.IOU_loss:
                metric_iou={"train/pupil_iou_loss": ciou_pupil,
                            "train/iris_iou_loss": ciou_iris,
                            "train/total_iou_loss": ciou_loss, 
                            }
                wandb.log(metric_iou)
            epoch_train_totalloss.append(total_loss.item())
            
            # replace with 
            if enum%30==0:
                print('[%d/%d, %5d/%d] train_total_loss: %.3f ' %
                    (epoch + 1, cfg.epochs, enum + 1, math.ceil(train_data_len / cfg.BATCH_SIZE), 
                    total_loss.item()))
        
        scheduler.step()

        print("Run Validation")

        seg_model.eval()
        epoch_val_totalloss = []
        epoch_val_innerloss = []
        epoch_val_outerloss = []
        segmentation = []
        regression = []
        # Evaluation
        count = 1
        with torch.no_grad():
            for enum, (inp, pupil_boundary, iris_boundary, cord, names) in enumerate(val_dataloader):
                inp = inp.to(cfg.device)
                pupil_boundary = pupil_boundary.to(cfg.device)
                iris_boundary = iris_boundary.to(cfg.device)
                cord = cord.to(cfg.device)

                pred_pupil, pred_iris, bbox = seg_model(inp)

                pupil_loss = criterion(pred_pupil, pupil_boundary)
                pupil_bce, pupil_dice = criterion.bce_loss, criterion.dice_loss
                
                iris_loss = criterion(pred_iris, iris_boundary)
                iris_bce, iris_dice = criterion.bce_loss, criterion.dice_loss
                
                if cfg.IoULoss:
                    ciou_pupil = complete_box_iou_loss(bbox[:, :4], cord[:, :4], reduction='mean') 
                    ciou_iris = complete_box_iou_loss(bbox[:, 4:], cord[:, 4:], reduction='mean')
                    ciou_loss = cfg.weight_pupil_ciou * ciou_pupil + cfg.weight_iris_ciou * ciou_iris 
                    bb_loss = cfg.weight_l1 * l1(bbox, cord) + cfg.weight_mse * mse(bbox, cord) + cfg.weight_ciou * ciou_loss
                else:    
                    bb_loss = cfg.weight_l1 * l1(bbox, cord) + cfg.weight_mse* mse(bbox, cord)
                
                metrics = {"val/pupil_bce": pupil_bce,
                           "val/pupil_dice": pupil_dice, 
                           "val/iris_bce": pupil_bce,
                           "val/iris_dice": pupil_dice, 
                           "val/total_loss": total_loss, 
                           "val/epoch": (enum + 1 + (n_steps_per_epoch * epoch)) / n_steps_per_epoch, 
                        }
                wandb.log(metrics)
                metric_bbox={ 
                            "val/bbox_reg_loss": bb_loss, 
                            }
                wandb.log(metric_bbox)
                if cfg.IOU_loss:
                    metric_iou={"val/pupil_iou_loss": ciou_pupil,
                                "val/iris_iou_loss": ciou_iris,
                                "val/total_iou_loss": ciou_loss, 
                                }
                    wandb.log(metric_iou)

                epoch_val_totalloss.append(total_loss.item())
                epoch_val_innerloss.append(pupil_bce.item() + pupil_dice.item())
                epoch_val_outerloss.append(iris_bce.item()+ iris_dice.item())
        
        # per epoch stats
        epoch_train_totalloss = np.mean(epoch_train_totalloss)

        epoch_val_totalloss = np.mean(epoch_val_totalloss)
        epoch_val_innerloss = np.mean(epoch_val_innerloss)
        epoch_val_outerloss = np.mean(epoch_val_outerloss)

        print('[%d/%d] train_totalloss: %.3f  val_totalloss: %.3f val_innerdice: %.3f val_outerdice: %.3f'
            % (epoch + 1, cfg.epochs, epoch_train_totalloss, epoch_val_totalloss, epoch_val_innerloss,
                epoch_val_outerloss))

        if epoch + 1 == cfg.epochs:
            torch.save(seg_model.state_dict(),
                    os.path.join(cfg.save_dir, 'epoch' + str(epoch + 1) + '.pth'))
            

        if epoch_val_innerloss > best_inner_dice:
            best_inner_dice = epoch_val_innerloss
            torch.save(seg_model.state_dict(),
                    os.path.join(cfg.save_dir, 'best_inner_dice.pth'))
            torch.save(seg_model.state_dict(),
                    os.path.join(cfg.save_dir, 'best_model.pth'))
        if epoch_val_outerloss > best_outer_dice:
            best_outer_dice = epoch_val_outerloss
            torch.save(seg_model.state_dict(),
                    os.path.join(cfg.save_dir, 'best_outer_dice.pth'))
            torch.save(seg_model.state_dict(),
                    os.path.join(cfg.save_dir, 'best_model.pth'))
            
def test(cfg:dict, model_path:str, tag:str):

    from metrics import compute_metric

    if cfg.IOU_loss == False:
        cfg.weight_ciou = 0

    create_folder(cfg.save_dir)
    
    val_data = CustomDatareader(image_folder=cfg.test_folder,
                                gt_path=cfg.test_gt, 
                                size=cfg.size, 
                                data_type = 'test'
                                )
    
    val_dataloader = DataLoader(dataset=val_data, batch_size=cfg.BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

    seg_model = UnetModel(encoder_name=cfg.enc_model).to(cfg.device) # TODO Default resnet34
    seg_model.load_state_dict(torch.load(model_path))
    val_data_len = val_data.length 

    criterion = DiceBCELoss(weight=(0.5, 0.5))   

    seg_model.eval()
    epoch_val_totalloss = []
    segmentation = []
    regression = []
    # Evaluation
    count = 1
    table = wandb.Table(columns=["Name","image", "pred", "Py_error", "PIoU", "Iy_error", "IIoU"])
    with torch.no_grad():
        for enum, (inp, pupil_boundary, iris_boundary, cord, names) in enumerate(val_dataloader):
            inp = inp.to(cfg.device)
            pupil_boundary = pupil_boundary.to(cfg.device)
            iris_boundary = iris_boundary.to(cfg.device)
            cord = cord.to(cfg.device)

            pred_pupil, pred_iris, bbox = seg_model(inp)

            # TODO TEST
            n = pupil_boundary.shape[0]
            for im_num in range(n):
                # Segmentation Model
                data, p_ellipse, i_ellipse, info = saveimage(pred_pupil[im_num], pred_iris[im_num], 
                                                            names[im_num], '/content/weights/{}_{}/'.format(cfg.model_name, tag))
                gt_file = names[im_num][:-4] 
                p_ellipse = p_ellipse.round(2)
                i_ellipse = i_ellipse.round(2)

                # Regression Model
                box = bbox[im_num].cpu().squeeze(0).detach().numpy() * 224

                segmentation.append([gt_file, list(data[:4]), list(p_ellipse), list(data[4:]), list(i_ellipse)])
                if cfg.IOU_loss == True:
                    pbox = bbox2ellipse(box[:4])
                    ibox = bbox2ellipse(box[4:8])
                    regression.append([gt_file, list(data[:4]), pbox, list(data[4:]), ibox])
                else:
                    box = box.round(2)  
                    regression.append([gt_file, list(data[:4]), list(box[:4]), list(data[4:]), list(box[4:])])
                # print(names[im_num])
                img_path = "/content/testing_set/images/"+names[im_num]
                img = imread(img_path)
                
                # Compute Metrics for Pupil and Iris
                px,py,piou = compute_metric(data[:4], p_ellipse)
                ix,iy,iiou = compute_metric(data[4:], i_ellipse)

                table.add_data(names[im_num], wandb.Image(img), 
                              wandb.Image(draw_output(img_path, info['p_ellipse'], info['i_ellipse'])),
                              100*py, 100*piou, 100*iy, 100*iiou)
                count += 1
        wandb.log({"predictions_table":table}, commit=False)
    save_csv(segmentation, cfg.save_dir+'/{}_seg.csv'.format(cfg.model_name)) # TODO Test
    save_csv(regression, cfg.save_dir+'/{}_reg.csv'.format(cfg.model_name)) # TODO Test
                                                                                                                                                       
    
