import torch
import pandas as pd
import csv
import pickle
# from nuscenes.nuscenes import NuScenes
from pyquaternion import Quaternion

# nusc = NuScenes(version='v1.0-trainval', dataroot='/home/gs/workspace/datasets/nuScenes/trainval', verbose=True)

# root_dir = './test.csv'

if __name__ == '__main__':
    # examples = pd.read_csv(root_dir, header=None)
    # prevs = [examples.iloc[i, 0] for i in range(len(examples))]
    # imgs = [examples.iloc[i, 1] for i in range(len(examples))]
    # nexts = [examples.iloc[i, 2] for i in range(len(examples))]
    # bev_gts = [examples.iloc[i, 3] for i in range(len(examples))]
    # RL_gts = [examples.iloc[i, 4] for i in range(len(examples))]
    # # /home/gs/workspace/datasets/nuScenes/seq_bev_dataset/scene-0005/cam_front/000_633212bb7ffa4953ac240019c9de2414.png,
    # print('len(imgs): ', len(imgs))
    # #for img in imgs:
    # for i in range(len(imgs)):
    #     print('{}:{}: '.format(i, imgs[i]))
    #     prev = prevs[i]
    #     img = imgs[i]
    #     next = nexts[i]
    #     bev_gt = bev_gts[i]
    #     RL_gt = RL_gts[i]

        # scene_num = imgs[i].split('/')[-3]
        # sample_toke = imgs[i].split('/')[-1].split('.')[0].split('_')[-1]

        # sample = nusc.get('sample', sample_toke)
        # samp = nusc.get('sample_data', sample['data']['CAM_FRONT'])

        # sens = nusc.get('calibrated_sensor', samp['calibrated_sensor_token'])

        # intrin = torch.Tensor(sens['camera_intrinsic']) # # type: torch.tensor
        # rot = torch.Tensor(Quaternion(sens['rotation']).rotation_matrix)
        # tran = torch.Tensor(sens['translation'])
        
        
        # info_row = [prev, img, next, bev_gt, RL_gt, intrin, rot, tran]
        
        # # 写入文件，可以保持数据类型
        # with open("./test_new.data", mode="ab") as fw:
        #     pickle.dump(info_row, fw)

        ###################################################################################################################
        # 读取
        # 更改路径 从/home/gs/workspace/datasets/nuScenes/seq_bev_dataset/ 改为 /workspace/data/nuScenes/seq_bev_dataset/

        examples = []
        with open('./val_new.data','rb') as f:  # old file
            while True:
                try:
                    list_data=pickle.load(f)
                    print("#")
                    prev_img, img, next_img, bev_gt, RL_gt, intrin, rot, tran = list_data
                    prev_img_new_path = '/workspace/data/nuScenes/seq_bev_dataset/' + '/'.join(prev_img.split('/')[-3:])
                    img_new_path = '/workspace/data/nuScenes/seq_bev_dataset/' + '/'.join(img.split('/')[-3:])
                    next_img_new_path = '/workspace/data/nuScenes/seq_bev_dataset/' + '/'.join(next_img.split('/')[-3:])
                    bev_gt_new_path = '/workspace/data/nuScenes/seq_bev_dataset/' + '/'.join(bev_gt.split('/')[-3:])
                    RL_gt_new_path = '/workspace/data/nuScenes/seq_bev_dataset/' + '/'.join(RL_gt.split('/')[-3:])
                    info_row = [prev_img_new_path, img_new_path, next_img_new_path, bev_gt_new_path, RL_gt_new_path, intrin, rot, tran]
                    with open("./val_new_1.data", mode="ab") as fw:  # new_file
                        pickle.dump(info_row, fw)
                    examples.append(list_data)
                    # break
                except EOFError:
                    break

        # print(len(examples))
        # print(examples[0])

        # print(list_data)

        ##########################################################################################################33
        # # 将train，val，test数据放在同一个文件中
        # examples = []
        # # data_files = ['./train_new.data', './val_new.data', './test_new.data']
        # data_files = ['./train_new.data', './val_new.data']
        # for data in data_files:
        #     with open(data,'rb') as f:  # old file
        #         while True:
        #             try:
        #                 # import pdb; pdb.set_trace()
        #                 list_data=pickle.load(f)
        #                 examples.append(list_data)
        #                 with open("./train_val.data", mode="ab") as fw:  # new_file
        #                     pickle.dump(list_data, fw)
        #             except EOFError:
        #                 break


        # 读一行测试
        examples = []
        with open('./train_new.data','rb') as f:
            while True:
                try:
                    list_data=pickle.load(f)
                    examples.append(list_data)
                except EOFError:
                    break
        print(len(examples))
        print(examples[0])

        print(list_data)

        # print(sens['camera_intrinsic'])
        # print(sens['rotation'])
        # print(sens['translation'])
        # break

##################################################################################################################################
### 计算类别权重 ###
# import os
# from tqdm import tqdm
# import numpy as np

# def calculate_weights_labels(dataloader, num_classes):
#     z = np.zeros((num_classes,))
#     tqdm_batch = tqdm(dataloader)
#     print('Calculating classes weights')
#     for sample in tqdm_batch:
#         (bev_labels, images, bev_images, img_names, scene_names,\
#              rots, trans, intrins, post_rots, post_trans) = sample
#         y = bev_labels.detach().cpu().numpy()
#         mask = (y>=0) & (y<num_classes)
#         labels = y[mask].astype(np.uint8)
#         count_l = np.bincount(labels, minlength=num_classes)
#         z += count_l
#     tqdm_batch.close()
#     total_frequency = np.sum(z)
#     class_weights = []
#     # 方法1 (两种方法计算的类别权重不同)
#     for frequency in z:
#         class_weight = 1/(np.log(1.02 + (frequency / total_frequency)))##这里是计算每个类别像素的权重
#         class_weights.append(class_weight)
#     ret = np.array(class_weights)
#     # # 方法2
#     # f_class_median=np.median(np.array(z)) # 计算类别频率中位数
#     # ret = f_class_median/np.array(z)
#     return ret, z