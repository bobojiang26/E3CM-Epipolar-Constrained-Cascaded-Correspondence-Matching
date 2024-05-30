import math
import numpy as np
import cv2
import argparse
import os
import glob
import yaml
import torch
import torch.hub
from torchvision import models, transforms
from utils import refinement_pose_ransac, display_canvas_matches


if __name__ == "__main__":

#   Parse command line arguments.
    parser = argparse.ArgumentParser(description='PyTorch Evaluation Demo.')
    parser.add_argument('--data_root', type=str,
        default='/media/zcb/Windows-SSD/testdata/megadepth_test_1500/Undistorted_SfM/0015/',
        help='Path to dataset')
    parser.add_argument('--find_fund_threshold', type=float, default=0.9,
        help='threshold in estimating pose')
    parser.add_argument('--outlier_rejection_threshold', type=float, default=0.01,
        help='outlier_rejection_threshold')
    parser.add_argument('--confidence_threshold', type=float, default=0.01,
        help='confidence_threshold')
    parser.add_argument('--refinement_pose', type=bool, default=True,
        help='if refinement_pose')
    parser.add_argument('--dataset_path', type=str,
        default='/media/zcb/Windows-SSD/hpatches-sequences-release/',
        help='Path to dataset')
    opt = parser.parse_args()


    approach = "DFM"

    detector = None
    matcher = None


    # from dfm import DeepFeatureMatcher
    from DFM_matcher import DeepFeatureMatcher

    with open("config.yml", "r") as configfile:
        config = yaml.safe_load(configfile)['configuration']
    fm = DeepFeatureMatcher(enable_two_stage = config['enable_two_stage'], model = config['model'],
                        ratio_th = config['ratio_th'], bidirectional = config['bidirectional'], )
    

    img1 = cv2.imread('/home/zcb/self_code_training/DFM_deep_networks/test_images/370516634_5c74ecd7c2_o.jpg')
    img2 = cv2.imread('/home/zcb/self_code_training/DFM_deep_networks/test_images/1357054806_316d0fe26a_o.jpg')
    
    img_H, img_W, _ = img1.shape

    # You can choose the backbone model by changing the 'model' in the config.yml
    if config['model'] != 'VGG19':
        img1 = cv2.resize(img1, None, fx=2, fy=2, interpolation= cv2.INTER_LINEAR)
        img2 = cv2.resize(img2, None, fx=2, fy=2, interpolation= cv2.INTER_LINEAR)


    _, _, points_A_list, points_B_list = fm.match_large_img(img1, img2)

    # hierarchical pose ransac
    if opt.refinement_pose:
        points_A, points_B = refinement_pose_ransac(points_A_list, points_B_list, opt.confidence_threshold)
    
    else:
        points_A = points_A_list[4]
        points_B = points_B_list[4]


    m_keypoints1 = points_A.T
    m_keypoints2 = points_B.T
    
    key = display_canvas_matches(img1, img2, 1, 1000, m_keypoints1, m_keypoints2)