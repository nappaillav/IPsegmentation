import wandb
import argparse, os
from trainer import train, test
from types import SimpleNamespace

default_config = SimpleNamespace(
        work_dir='/content/',
        train_folder='/content/training_set/images/',
        train_gt = '/content/training_set/groundtruth/',
        test_folder = '/content/testing_set/images/',
        test_gt = '/content/testing_set/groundtruth/',
        save_dir = '/content/weights',
        enc_model = 'resnet18',
        model_name = 'Unet_DB_R18',
        device = 'cuda',
        BATCH_SIZE = 16,
        lr_max = 0.0002,
        L2 = 0.0001,
        epochs = 5,
        IOU_loss = False,  
        size = (224, 224),
        weight_pupil_ciou = 1,
        weight_iris_ciou = 0.5,
        weight_l1 = 1,
        weight_mse = 1,
        weight_ciou = 0.5, #[0, 0.5, 1],  # if bbox_loss is true
        weight_pdice = 1,
        weight_idice = 1,
        weight_bbox = 0.5,
    )

def parse_args():
    "Overriding default argments"
    argparser = argparse.ArgumentParser(description='Process hyper-parameters')
    argparser.add_argument('--work_dir', type=str, default='/content/', help='SRC directory')
    argparser.add_argument('--train_folder', type=str, default='/content/training_set/images/', help='Train images directory')
    argparser.add_argument('--train_gt', type=str, default='/content/training_set/groundtruth/', help='Train groundtruth directory')
    argparser.add_argument('--test_folder', type=str, default='/content/testing_set/images/', help='Test images directory')
    argparser.add_argument('--test_gt', type=str, default='/content/testing_set/groundtruth/', help='Test groundtruth directory')
    argparser.add_argument('--save_dir', type=str, default='/content/weights', help='Save Results in this directory')
    argparser.add_argument('--enc_model', type=str, default='resnet18', help='Encoder Model')
    argparser.add_argument('--model_name', type=str, default='Unet_DB_R18', help='Model name')
    argparser.add_argument('--device', type=str, default='cuda', help='Device')
    argparser.add_argument('--batchsize', type=int, default=8, help='Batch Size')
    argparser.add_argument('--lr_max', type=float, default=0.0002, help='Maximum Learning rate')
    argparser.add_argument('--L2', type=float, default=0.0001, help='L2 weight')
    argparser.add_argument('--epochs', type=int, default=5, help='Total number of epochs')
    argparser.add_argument('--IOU_loss', type=bool, default=False, help='To include IOU loss')
    argparser.add_argument('--size', type=int, default=224, help='Size of the image')
    argparser.add_argument('--weight_pupil_ciou', type=float, default=1.0, help='weight for pupil iou loss')
    argparser.add_argument('--weight_iris_ciou', type=float, default=0.5, help='weight for iris iou loss')
    argparser.add_argument('--weight_l1', type=float, default=1.0, help='weight for L1 loss')
    argparser.add_argument('--weight_mse', type=float, default=1.0, help='weight for MSE loss')
    argparser.add_argument('--weight_ciou', type=float, default=0.5, help='weight for cIoU loss')
    argparser.add_argument('--weight_pdice', type=float, default=1.0, help='weight for pupil dice loss')
    argparser.add_argument('--weight_idice', type=float, default=1.0, help='weight for iris dice loss')
    argparser.add_argument('--weight_bbox', type=float, default=0.5, help='weight for bounding box loss')
    args = argparser.parse_args()
    vars(default_config).update(vars(args))
    return

if __name__ == "__main__":
    
    
    parse_args()

    wandb.init(
        project='IPSegmentation_debug',
        name=default_config.model_name,
        config=default_config
        )
    # trainer code
    train(wandb.config)

    # test on test set 
    test(wandb.config, '{}/best_model.pth'.format(default_config.save_dir), 'pupil_model')

    wandb.finish()