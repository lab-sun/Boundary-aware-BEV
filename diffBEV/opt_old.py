import argparse
import os

from parso import parse

from easydict import EasyDict as edict

def get_args():
    parser = argparse.ArgumentParser(description="DiffBEV options")
    # dataset相关
    parser.add_argument("--dataroot", type=str, default="./data/nuScenes",
                        help="Path to the nuscenes csv files")
    parser.add_argument("--train_csv", type=str, default="./data/nuScenes/train.csv",
                        help="Path to the train csv file")
    parser.add_argument("--val_csv", type=str, default="./data/nuScenes/val.csv",
                        help="Path to the val csv file")
    parser.add_argument("--test_csv", type=str, default="./data/nuScenes/test.csv",
                        help="Path to the test csv file")
    parser.add_argument("--ae_model_path", type=str, default="./pretrain_weights/ae",
                        help="pretrained weight for autoencoder")  # 将训练好的ae模型放在该路径下
    label_list = ['background', 'drivable_area', 'ped_crossing', 'walkway', 'movable_object', 'vehicle', 'predestrian']
    parser.add_argument("--label_list", type=dict, default=label_list,
                        help="label_list")



    # lss相关
    H=900
    W=1600
    map_extents = [-15., 1., 15., 31.]
    map_resolution = 0.2
    resize_lim=(0.193, 0.225)
    #final_dim=(128, 352)
    #final_dim=(128, 256)
    final_dim=(256, 512)
    bot_pct_lim=(0.0, 0.22)
    rot_lim=(-5.4, 5.4)
    rand_flip=True
    ncams=1
    max_grad_norm=5.0
    pos_weight=2.13
    data_aug_conf = {
            'resize_lim': resize_lim,
            'final_dim': final_dim,
            'rot_lim': rot_lim,
            'H': H, 'W': W,
            'rand_flip': rand_flip,
            'bot_pct_lim': bot_pct_lim,
            'cams': ['CAM_FRONT'],
            'Ncams': ncams,
            'map_extents': map_extents,
            'map_resolution': map_resolution,
        }
    parser.add_argument("--data_aug_conf", type=dict, default=data_aug_conf,
                        help="frustum boundary for lss")
    
    xbound=[-37.5, 37.5, 0.5]
    ybound=[-37.5, 37.5, 0.5]
    zbound=[-10.0, 10.0, 20.0]
    dbound=[4.0, 85.0, 1.0]
    grid_conf = {
        'xbound': xbound,
        'ybound': ybound,
        'zbound': zbound,
        'dbound': dbound,
    }
    parser.add_argument("--grid_conf", type=dict, default=grid_conf,
                        help="frustum boundary for lss")  

    parser.add_argument("--lss_lr", type=float, default=1e-3, # 1e-3
                        help="lss learning rate") # lss_lr
    parser.add_argument("--text_logger_path", type=str, default='./logs/', # 1e-3
                        help="path to log file") # lss_lr
    parser.add_argument("--n_epochs_lss", type=int, default=50,
                        help="Number of epoch")
        

    
    # save相关
    parser.add_argument("--save_path", type=str, default="./pretrain_weights/",
                        help="path to save models")
    parser.add_argument("--model_name", type=str, default="diffBEV",
                        help="Model Name with specifications")
    parser.add_argument("--if_save_img", type=bool, default=True,
                        help="Whether to save test img")
    parser.add_argument("--save_img_path", type=str, default="./saved_img/",
                        help="path to save test images")
    parser.add_argument("--log_path", type=str, default="./logs/",
                        help="path to log info")
    parser.add_argument("--log_frequency", type=int, default=50,
                        help="Log files every x epochs")
    
    # 训练相关
    parser.add_argument("--num_class", type=int, default=7,
                        help="Number of classes")
    parser.add_argument("--loss_type", type=str, default="focal",
                        choices=['ce', 'focal'],
                        help="loss type of bev prediction")
    parser.add_argument('--dice_weight', type=float, default=1.0,
                        help='use ce loss and dice loss to calculate the bev loss at the same time, \
                            the weight of dice loss when add those two losses together')
    parser.add_argument("--if_BoundaryLoss", type=bool, default=True,
                        help="Whether to save test img")
    parser.add_argument('--boundaryLoss_gamma', type=float, default=1.5,
                        help='gamma for boundary loss weight calucation')
    
    parser.add_argument("--batch_size", type=int, default=8,
                        help="Mini-Batch size")
    parser.add_argument("--test_batch_size", type=int, default=2,
                        help="Mini-Batch size")
    parser.add_argument("--lr", type=float, default=5e-4, 
                        help="learning rate") # basic_lr
    parser.add_argument("--num_epochs", type=int, default=50,
                        help="Max number of training epochs")
    parser.add_argument("--sample_interval", type=int, default=1,
                        help="the epoch to sample the whole dataset")

    configs = edict(vars(parser.parse_args()))
    config_list = []
    for key in configs.keys():
        config_list.append(key)
        config_list.append(configs[key])

    return configs
    