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
from PIL import Image

def to_homogeneous(points):
    return torch.cat([points, torch.ones_like(points[:, :1])],dim=1)

def mnn_matcher(descriptors_a, descriptors_b):
    device = descriptors_a.device
    sim = descriptors_a @ descriptors_b.t()
    nn12 = torch.max(sim, dim=1)[1]
    nn21 = torch.max(sim, dim=0)[1]
    ids1 = torch.arange(0, sim.shape[0], device=device)
    mask = (ids1 == nn21[nn12])
    matches = torch.stack([ids1[mask], nn12[mask]])
    return matches.t().data.cpu().numpy()

def get_H_matrix(h_file_path):
    file = open(h_file_path, 'r')
    h_split = file.read().split()
    H = np.zeros((3, 3))
    for j in range(3):
        for i in range(3):
            H[j][i] = h_split[j * 3 + i]
    return H


def get_gt_H_matrix(data_path, id1, id2):
    if id1 == 1:
        h_filename = "H_" + str(id1) + "_" + str(id2)
        return get_H_matrix(data_path + "/" + h_filename)
    else:
        h_file1 = "H_1_" + str(id1)
        h_file2 = "H_1_" + str(id2)
        H1 = get_H_matrix(data_path + "/" + h_file1)
        H2 = get_H_matrix(data_path + "/" + h_file2)
        return np.linalg.inv(H1)@H2

def compute_pose(points_A, points_B):

    Fundmat, mask = cv2.findFundamentalMat(points_A.T, points_B.T, method=cv2.FM_RANSAC,ransacReprojThreshold=1, confidence=0.99)

    return Fundmat
    
def outlier_rejection_confidence(points_A, points_B, confidence_threshold, Fundmat):

    if Fundmat.shape[0] == 9:
        Fundmat = Fundmat[:3,:]

    keypoints_1 = points_A.T
    keypoints_2 = points_B.T

    keypoints_num = keypoints_1.shape[0]

    residules = (to_homogeneous(torch.from_numpy(keypoints_1)).numpy() @ Fundmat @ to_homogeneous(torch.from_numpy(keypoints_2)).numpy().T).diagonal()
    
    rank_index = np.argsort(residules)

    confidences = 13 * np.log10(abs(rank_index / (keypoints_num * residules)))
    # rank = np.argsort(confidences)
    # confidences_rank = confidences[rank]
    # print(confidences_rank[int(keypoints_num*0.3)])
    satisfied_keypoints_index = np.where(confidences > confidence_threshold)
        
    refined_keypoints_1 = keypoints_1[satisfied_keypoints_index]
    refined_keypoints_2 = keypoints_2[satisfied_keypoints_index]
    
    return refined_keypoints_1.T, refined_keypoints_2.T

def outlier_rejection(points_A, points_B, threshold,Fundmat ):

    if Fundmat.shape[0] != 3:
        Fundmat = Fundmat[:3,:]

    points_A_refined = []
    points_B_refined = []

    kpts0 = points_A.T
    kpts1 = points_B.T

    points_A_homo = to_homogeneous(torch.from_numpy(kpts0))
    points_B_homo = to_homogeneous(torch.from_numpy(kpts1))

    # loss_list = np.array([0.0])

    # for i in range(points_A.T.shape[0]):
    #     loss = points_B_homo[i,:].resize(1,3) @ torch.from_numpy(Fundmat) @ points_A_homo[i,:].resize(3,1)
    #     loss_list = np.append(loss_list, loss)
    
    # loss_list = np.sort(loss_list)
    # loss_average = np.mean(loss_list)
    # loss_thre = loss_list[int(len(loss_list)*threshold)]


    for i in range(points_A.T.shape[0]):
        loss = points_B_homo[i,:].resize(1,3) @ torch.from_numpy(Fundmat) @ points_A_homo[i,:].resize(3,1)
        if loss < threshold and loss > (0 -threshold):
            points_A_refined.append(torch.from_numpy(points_A.T)[i,:])
            points_B_refined.append(torch.from_numpy(points_B.T)[i,:])
    
    if len(points_A_refined) == 0:
        return points_A, points_B
  
    points_A_refined = torch.stack(points_A_refined) 
    points_B_refined = torch.stack(points_B_refined) 

    return points_A_refined.t().numpy(), points_B_refined.t().numpy()


def refinement_pose_ransac(points_A_list, points_B_list, confidence_threshold):
    if points_A_list[0].shape[1] < 8:
        points_A = points_A_list[4]
        points_B = points_B_list[4]
    else :
        Fundmat = compute_pose(points_A_list[0], points_B_list[0])

        for i in range(1,5):
            if type(Fundmat) == type(None):
                points_A = points_A_list[4]
                points_B = points_B_list[4]
            else:
                points_A = points_A_list[i] #shape: 2 * num
                points_B = points_B_list[i]
                if points_A.shape[1] < 8:
                    points_A = points_A_list[4]
                    points_B = points_B_list[4]
                else :
                    points_A, points_B = outlier_rejection(points_A, points_B, confidence_threshold, Fundmat)      
                    Fundmat = compute_pose(points_A, points_B)
   
    return points_A, points_B


def display_canvas_matches(img1, img2, num, wait_time, m_pts1, m_pts2):
    img1_height = img1.shape[0]
    img1_width = img1.shape[1]
    img2_height = img2.shape[0]
    img2_width = img2.shape[1]
    canvas = np.zeros((max(img1_height, img2_height), img1_width + img2_width, 3), np.uint8)
    if len(img1.shape) == 3:
        canvas[:img1_height, :img1_width] = img1
        canvas[:img2_height, img1_width:] = img2
    else:
        canvas[:img1_height, :img1_width, 0] = img1
        canvas[:img1_height, :img1_width, 1] = img1
        canvas[:img1_height, :img1_width, 2] = img1
        canvas[:img2_height, img1_width:, 0] = img2
        canvas[:img2_height, img1_width:, 1] = img2
        canvas[:img2_height, img1_width:, 2] = img2
    
    
#   draw circle and line with confidence(heat)
    for i in range(m_pts1.shape[0]):
        kp1 = (int(m_pts1[i][0]),  int(m_pts1[i][1]))
        kp2 = (int(m_pts2[i][0] + img1_width),  int(m_pts2[i][1]))
    #   confidence of keypoint generated by superpoint
    #   attention : classical method has no confidence
        # conf_kp1 = m_pts1[i][2]
        # color_kp1_n = color_bar[0][int(255*conf_kp1)]
        # color_kp1 = (int(color_kp1_n[0]), int(color_kp1_n[1]), int(color_kp1_n[2]))
        # cv2.circle(canvas, kp1, 1, color_kp1, -1)

        # conf_kp2 = pts2[int(m[1])][2]
        # color_kp2_n = color_bar[0][int(255*conf_kp2)]
        # color_kp2 = (int(color_kp2_n[0]), int(color_kp2_n[1]), int(color_kp2_n[2]))
        # cv2.circle(canvas, kp2, 1, color_kp2, -1)

    #   confidence of matches generated by matcher
        # color_n = color_bar[0][int(255*m_confs[i])]
        #color = (int(color_n[0]), int(color_n[1]), int(color_n[2]))

        
        # if distances[i] < threshold:
        #     color = (0, 255, 0)
        # if distances[i] >= threshold:
        #     color = (0, 0, 255)
        color = (0, 255, 0)
        cv2.line(canvas, kp1, kp2, color, 2)

    cv2.imshow("canvas", canvas)
    key = cv2.waitKey(wait_time)
    cv2.imwrite('canvas_'+str(num)+'.jpg', canvas)
    
    return key 

#   error measure
def print_f_norm(H_cal, H_gt):
    if H_cal is None:
        print("\033[0;31m", "ERROR : H_cal is None", "\033[0m")
    else:
        I = np.eye(3)
        print("||inv(H_cal)@H_gt-I||", np.linalg.norm(np.linalg.inv(H_cal)@H_gt - I))


def get_MMA(H_gt, m_pts1, m_pts2, thres):
    t = thres*thres
    N = len(m_pts1)
    sum_value = 0
    for i in range(N):
        new_pt = H_gt @ np.array([m_pts1[i][0], m_pts1[i][1], 1])
        new_pt /= new_pt[2]
    #   We don't consider the last line now
        du = new_pt[0] - m_pts2[i][0]
        dv = new_pt[1] - m_pts2[i][1]
        L2 = du*du + dv*dv
        if L2 < t:
            sum_value = sum_value + 1
    if sum_value > 0:
        return sum_value/N
    else:
        return 0.0