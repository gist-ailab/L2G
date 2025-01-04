import os
import glob

import h5py
import json
import numpy as np
import torch

import copy
import open3d as o3d

from l2g_core.utils.grasp_utils import getOrientation2
# split the grasps into train and test

def split_grasps(root, split='train'):
    
    train_list_root = f'DA2/processed_total_{split}_json/grasps'
    train_list = glob.glob(os.path.join(root, train_list_root, '*.json'))
    train_list = [os.path.basename(f) for f in train_list]
    train_list = [os.path.splitext(f)[0] for f in train_list]
    
    train_split_folder = os.path.join(root, f'DA2/{split}_split')
    if not os.path.exists(train_split_folder):
        os.makedirs(train_split_folder)
    
    origin_folder = os.path.join(root, 'DA2/grasps')
    origin_data_list = glob.glob(os.path.join(origin_folder, '*.h5'))

    for data in origin_data_list:
        file_name = os.path.splitext(os.path.basename(data))[0]
        if file_name in train_list:
            print(f"Moving {data} to {split}_split")
            os.rename(data, data.replace('grasps', f'{split}_split'))


def main():
    # train and test split list
    root = 'data'
    # split_grasps(root, 'train')
    da2_root = 'DA2'
    train_split = 'train_split'
    
    train_path = os.path.join(root, da2_root, train_split)
    train_data_list = glob.glob(os.path.join(train_path, '*.h5'))
    for data in train_data_list:
        h5_data = h5py.File(data, 'r')
        grasp_transform = np.asarray(h5_data['grasps/transforms']) # (2001, 2, 4, 4)
        grasp_angle = np.asarray(h5_data['grasps/angle']) # (2001, 2)
        grasp_points = np.asarray(h5_data['grasps/grasp_points']) # (2001, 4, 3)
        grasp_center = np.asarray(h5_data['grasps/center']) # (2001, 2, 3)
        # print(torch.tensor(grasp_points[0][0]).unsqueeze(0).shape)
        # print(torch.tensor(grasp_center[0][0]).unsqueeze(0).shape)
        # print(torch.tensor(grasp_angle[0][0]).unsqueeze(0).shape)
        
        grasp_matrix = getOrientation2(torch.tensor(grasp_points[0][0], dtype=torch.float32).unsqueeze(0), 
                                       torch.tensor(grasp_center[0][0], dtype=torch.float32).unsqueeze(0), 
                                       torch.tensor(grasp_angle[0][0], dtype=torch.float32).unsqueeze(0))
        print(grasp_matrix)
        print(grasp_transform[0])
        
        # visualize the grasps, orignal and calculated
        control_points = [[0,0,-0.03375], [0.0425, 0, 0],
                      [-0.0425, 0, 0], [0.0425, 0, 0.0675],
                      [-0.0425, 0, 0.0675]]
        control_points = np.asarray(control_points, dtype=np.float32)
        # move control points -0.02 to z axis
        # control_points[:, 2] -= 0.02    
        mid_point = 0.5*(control_points[1] + control_points[2])
        tmp_control_points = []
        tmp_control_points.append(control_points[0])
        tmp_control_points.append(mid_point)
        tmp_control_points.append(control_points[1])
        tmp_control_points.append(control_points[3])
        tmp_control_points.append(control_points[1])
        tmp_control_points.append(control_points[2])
        tmp_control_points.append(control_points[4])
        control_points = np.asarray(tmp_control_points, dtype=np.float32) # (7, 3)
        control_points[2:, 0] = np.sign(control_points[2:, 0]) * 0.085/2
        ones = np.ones((len(control_points), 1))
        control_points = np.concatenate((control_points, ones), -1) # (7, 4)
        control_points_tmp = copy.deepcopy(control_points)
        control_points_tmp[:, 2] = -control_points[:, 1]
        control_points_tmp[:, 1] = control_points[:, 2]
        control_points = control_points_tmp # (7, 4)
        
        grasp_ori = np.matmul(control_points, np.transpose(grasp_transform[0][0], (1, 0)))[:, :3] # (7, 3)
        grasp_mat_cal = np.concatenate((grasp_matrix[0].numpy(), grasp_transform[0][0][:3, 3:]), axis=1) # (3, 4)
        grasp_mat_cal = np.concatenate((grasp_mat_cal, np.array([[0, 0, 0, 1]])), 0) # (4, 4)
        grasp_cal = np.matmul(control_points, np.transpose(grasp_mat_cal, (1, 0)))[:, :3] # (7, 3)
        
        



if __name__ == '__main__':
    main()