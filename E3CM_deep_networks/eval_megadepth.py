import numpy as np
import cv2
import argparse
import random
import os
import glob
import yaml
import torch
import torch.hub
from torchvision import models, transforms
from utils import to_homogeneous, refinement_pose_ransac, display_canvas_matches


def error_auc(errors, thresholds):
    """
    Args:
        errors (list): [N,]
        thresholds (list)
    """
    errors = [0] + sorted(list(errors))
    recall = list(np.linspace(0, 1, len(errors)))

    aucs = []
    thresholds = [5, 10, 20]
    for thr in thresholds:
        last_index = np.searchsorted(errors, thr)
        y = recall[:last_index] + [recall[last_index-1]]
        x = errors[:last_index] + [thr]
        aucs.append(np.trapz(y, x) / thr)

    return {f'auc@{t}': auc for t, auc in zip(thresholds, aucs)}

def estimate_pose(kpts0, kpts1, K0, K1, thresh, conf=0.99999):
    if len(kpts0) < 5:
        return None
    # normalize keypoints
    kpts0 = (kpts0 - K0[[0, 1], [2, 2]][None]) / K0[[0, 1], [0, 1]][None]
    kpts1 = (kpts1 - K1[[0, 1], [2, 2]][None]) / K1[[0, 1], [0, 1]][None]

    # normalize ransac threshold
    ransac_thr = thresh / np.mean([K0[0, 0], K0[1, 1], K1[0, 0], K1[1, 1]])

    # compute pose with cv2
    E, mask = cv2.findEssentialMat(
        kpts0, kpts1, np.eye(3), threshold=ransac_thr, prob=conf, method=cv2.RANSAC)
    if E is None:
        print("\nE is None while trying to recover pose.\n")
        return None

    # recover pose from E
    best_num_inliers = 0
    ret = None
    for _E in np.split(E, len(E) / 3):
        n, R, t, _ = cv2.recoverPose(_E, kpts0, kpts1, np.eye(3), 1e9, mask=mask)
        if n > best_num_inliers:
            ret = (R, t[:, 0], mask.ravel() > 0)
            best_num_inliers = n

    return ret



def relative_pose_error(R_groundtruth, t_groundtruth, R, t, ignore_gt_t_thr=0.0):
    # angle error between 2 vectors
    t_gt = t_groundtruth
    n = np.linalg.norm(t) * np.linalg.norm(t_gt)
    t_err = np.rad2deg(np.arccos(np.clip(np.dot(t, t_gt) / n, -1.0, 1.0)))
    t_err = np.minimum(t_err, 180 - t_err)  # handle E ambiguity
    if np.linalg.norm(t_gt) < ignore_gt_t_thr:  # pure rotation is challenging
        t_err = 0

    # angle error between 2 rotation matrices
    R_gt = R_groundtruth
    cos = (np.trace(np.dot(R.T, R_gt)) - 1) / 2
    cos = np.clip(cos, -1., 1.)  # handle numercial errors
    R_err = np.rad2deg(np.abs(np.arccos(cos)))

    return t_err, R_err

def q_to_r(q):
    r = np.eye(3)
    r = np.eye(3)
    r[0][0] = 1-2*q[2]*q[2]-2*q[3]*q[3]
    r[0][1] = 2*q[1]*q[2] + 2*q[0]*q[3]
    r[0][2] = 2*q[1]*q[3] - 2*q[0]*q[2]
    r[1][0] = 2*q[1]*q[2] - 2*q[0]*q[3]
    r[1][1] = 1-2*q[1]*q[1]-2*q[3]*q[3]
    r[1][2] = 2*q[2]*q[3] + 2*q[0]*q[1]
    r[2][0] = 2*q[1]*q[3] + 2*q[0]*q[2]
    r[2][1] = 2*q[2]*q[3] - 2*q[0]*q[1]
    r[2][2] = 1-2*q[2]*q[2]-2*q[1]*q[1]
    return r

def translation_mat(translation):
    tran_mat = np.eye(3)
    tran_mat[0][0] = 0
    tran_mat[0][1] = -translation[2]
    tran_mat[0][2] = translation[1]
    tran_mat[1][0] = translation[2]
    tran_mat[1][1] = 0
    tran_mat[1][2] = -translation[0]
    tran_mat[2][0] = -translation[1]
    tran_mat[2][1] = translation[0]
    tran_mat[2][2] = 0
    return tran_mat



def read_images_txt(data_path, img1_name, img2_name):

    data = {}

    q1 = np.array([1.0,1.0,1.0,1.0])
    q2 = np.array([1.0,1.0,1.0,1.0])
    t1 = np.array([1.0,1.0,1.0])
    t2 = np.array([1.0,1.0,1.0])
    points_3d_img1 = {}
    points_3d_img2 = {}
    point_3d_id = np.array([1.0])

    line_list = []
    with open(data_path, "r") as f:
        
        for line in f.readlines():

            line_list.append(line)
            
    for i in range(2,len(line_list)//2):

        line_element = line_list[2*i].split()

        if line_element[9] == img1_name:
            q1[0] = line_element[1]
            q1[1] = line_element[2]
            q1[2] = line_element[3]
            q1[3] = line_element[4]
            t1[0] = line_element[5]
            t1[1] = line_element[6]
            t1[2] = line_element[7]
            img1_camera = line_element[8]
        
            points3d_line_element = line_list[2*i+1].split()
            for j in range(len(points3d_line_element)//3):
                if points3d_line_element[j*3+2] != '-1':
                    point_3d_id[0] = points3d_line_element[j*3+2]
                    points_3d_img1[points3d_line_element[j*3+2]] = point_3d_id
        
        if line_element[9] == img2_name:
            q2[0] = line_element[1]
            q2[1] = line_element[2]
            q2[2] = line_element[3]
            q2[3] = line_element[4]
            t2[0] = line_element[5]
            t2[1] = line_element[6]
            t2[2] = line_element[7]
            img2_camera = line_element[8]
        
            points3d_line_element = line_list[2*i+1].split()
            for j in range(len(points3d_line_element)//3):
                if points3d_line_element[j*3+2] != '-1':
                    point_3d_id[0] = points3d_line_element[j*3+2]
                    points_3d_img2[points3d_line_element[j*3+2]] = point_3d_id
            
    R_1 = q_to_r(q1)
    R_2 = q_to_r(q2)
    R = np.linalg.inv(R_2) @ R_1
    T = t1-t2
    data['rotation'] = R
    data['translation'] = T
    data['img1_camera'] = img1_camera # camera id
    data['img2_camera'] = img2_camera #camera id
    data['points_3d_img1'] = points_3d_img1 #list
    data['points_3d_img2'] = points_3d_img2 #list

    return data

def read_intrinsics(data_path, img1_camera, img2_camera):
    K_1 = np.eye(3)
    K_2 = np.eye(3)
    with open(data_path, "r") as f:
        for line in f.readlines():
            line_element = line.split()

            if line_element[0] == img1_camera:
                K_1[0][0] = line_element[4]
                K_1[1][1] = line_element[5]
                K_1[0][2] = line_element[6]
                K_1[1][2] = line_element[7]
            
            if line_element[0] == img2_camera:
                K_2[0][0] = line_element[4]
                K_2[1][1] = line_element[5]
                K_2[0][2] = line_element[6]
                K_2[1][2] = line_element[7]
    
    return K_1, K_2

def read_pair_data(data_path_list):
    data={
        'img1_list':[],
        'img2_list':[],
        'rotation':[],
        'translation':[],
        'intrinsics1':[],
        'intrinsics2':[]
    }
    for data_path in data_path_list:
        img1_num_list = []
        img2_num_list = []
        dataset = np.load(data_path, allow_pickle=True)
        for i in range(len(dataset['pair_infos'])):
            img1_num = dataset['pair_infos'][i][0][0]
            img2_num = dataset['pair_infos'][i][0][1]

            img1_num_list.append(img1_num)
            img2_num_list.append(img2_num)

        for i in range(len(img1_num_list)):
            data['img1_list'].append(dataset['image_paths'][img1_num_list[i]])
            data['img2_list'].append(dataset['image_paths'][img2_num_list[i]])
            T0 = dataset['poses'][img1_num_list[i]]
            T1 = dataset['poses'][img2_num_list[i]]
            T0_1 = np.matmul(T1, np.linalg.inv(T0))
            rotation = T0_1[:3,:3]
            transaltion = T0_1[:3,3:4].reshape(3)
            data['rotation'].append(rotation)
            data['translation'].append(transaltion)
            data['intrinsics1'].append(dataset['intrinsics'][img1_num_list[i]])
            data['intrinsics2'].append(dataset['intrinsics'][img2_num_list[i]])

    
    return data

def symmetric_epipolar_distance(pts0, pts1, E, K0, K1):
    """Squared symmetric epipolar distance.
    This can be seen as a biased estimation of the reprojection error.
    Args:
        pts0 (torch.Tensor): [N, 2]
        E (torch.Tensor): [3, 3]
    """
    pts0 = (pts0 - K0[[0, 1], [2, 2]][None]) / K0[[0, 1], [0, 1]][None]
    pts1 = (pts1 - K1[[0, 1], [2, 2]][None]) / K1[[0, 1], [0, 1]][None]
    pts0 = to_homogeneous(pts0)
    pts1 = to_homogeneous(pts1)

    Ep0 = pts0 @ E.T  # [N, 3]
    p1Ep0 = torch.sum(pts1 * Ep0, -1)  # [N,]
    Etp1 = pts1 @ E  # [N, 3]

    d = p1Ep0**2 * (1.0 / (Ep0[:, 0]**2 + Ep0[:, 1]**2) + 1.0 / (Etp1[:, 0]**2 + Etp1[:, 1]**2))  # N
    return d


def epidist_prec(errors, thresholds, ret_dict=False):
    precs = []
    for thr in thresholds:
        prec_ = []
        for errs in errors:
            correct_mask = errs < thr
            prec_.append(np.mean(correct_mask.numpy()) if len(correct_mask) > 0 else 0)
        precs.append(np.mean(prec_) if len(prec_) > 0 else 0)
    if ret_dict:
        return {f'prec@{t:.0e}': prec for t, prec in zip(thresholds, precs)}
    else:
        return precs


            
    
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='PyTorch Evaluation Demo.')
    parser.add_argument('--data_root', type=str,
        default='/media/zcb/zcb/testdata/megadepth_test_1500/',
        help='Path to dataset')
    parser.add_argument('--estimate_pose_threshold', type=float, default=0.5,
        help='threshold in estimating pose')
    parser.add_argument('--outlier_rejection_threshold', type=float, default=1e-4,
        help='outlier_rejection_threshold')
    parser.add_argument('--refinement_pose', type=bool, default=False,
        help='if refinement_pose')
    parser.add_argument('--pair_info_list', type=list, default=['megadepth_test_1500_scene_info/0015_0.1_0.3.npz','megadepth_test_1500_scene_info/0015_0.3_0.5.npz',
    'megadepth_test_1500_scene_info/0022_0.1_0.3.npz','megadepth_test_1500_scene_info/0022_0.3_0.5.npz','megadepth_test_1500_scene_info/0022_0.5_0.7.npz'],
        help='pair_info_list')
    opt = parser.parse_args()
    
    from DFM_matcher import DeepFeatureMatcher

    with open("config.yml", "r") as configfile:
        config = yaml.safe_load(configfile)['configuration']
    fm = DeepFeatureMatcher(enable_two_stage = config['enable_two_stage'], model = config['model'], 
                        ratio_th = config['ratio_th'], bidirectional = config['bidirectional'], )
    
    
    sift = cv2.SIFT_create()

    data = read_pair_data(opt.pair_info_list)

    results={'R_errs':[],
              't_errs':[],
              'inliers':[],
              'epi_errs':[]}

    # test_pairs_count = 0

    # while(test_pairs_count<1500): 
    with torch.no_grad():
        for i in range(len(data['img1_list'])):

            rotation = data['rotation'][i]
            translation = data['translation'][i]

            E = translation_mat(translation) @ rotation

            k1 = data['intrinsics1'][i]
            k2 = data['intrinsics2'][i]

            img1_path = opt.data_root + data['img1_list'][i]
            img2_path = opt.data_root + data['img2_list'][i]

            img1 = cv2.imread(img1_path)
            img2 = cv2.imread(img2_path)

            # if resize images
            # img1_h, img1_w, _ = img1.shape
            # img2_h, img2_w ,_ = img2.shape

            # f_1 = 1200/max(img1_h, img1_w)
            # f_2 = 1200/max(img2_h, img2_w)

            # img1 = cv2.resize(img1, None, fx=f_1, fy=f_1, interpolation= cv2.INTER_LINEAR)
            # img2 = cv2.resize(img2, None, fx=f_2, fy=f_2, interpolation= cv2.INTER_LINEAR)

            # k1 *= f_1
            # k1[2][2] = 1
            # k2 *= f_2
            # k2[2][2] = 1

            if config['model'] != 'VGG19':
                # img1 = cv2.pyrUp(img1)
                # img2 = cv2.pyrUp(img2)
                img1 = cv2.resize(img1, None, fx=2, fy=2, interpolation= cv2.INTER_LINEAR)
                img2 = cv2.resize(img2, None, fx=2, fy=2, interpolation= cv2.INTER_LINEAR)

            _, _, points_A_list, points_B_list = fm.match_small_img(img1, img2)

            
            if opt.refinement_pose:
                points_A, points_B = refinement_pose_ransac(points_A_list, points_B_list, opt.outlier_rejection_threshold, opt.estimate_pose_threshold, k1, k2)
            
            else:
                points_A = points_A_list[3]
                points_B = points_B_list[3]

        
            m_keypoints1 = points_A.T
            m_keypoints2 = points_B.T
            
            # kp1, des1 = sift.detectAndCompute(img1, None)
            # kp2, des2 = sift.detectAndCompute(img2, None)

            # bf = cv2.BFMatcher_create(cv2.NORM_L1, crossCheck=False)
            # matches = bf.match(des1, des2)

            # m_keypoints1 = np.ones((len(matches),2))
            # m_keypoints2 = np.ones((len(matches),2))

            # for k in range(len(matches)):
            #     m_keypoints1[k][0] = kp1[matches[k].queryIdx].pt[0]
            #     m_keypoints1[k][1] = kp1[matches[k].queryIdx].pt[1]
            #     m_keypoints2[k][0] = kp2[matches[k].trainIdx].pt[0]
            #     m_keypoints2[k][1] = kp2[matches[k].trainIdx].pt[1]
        

            distances = symmetric_epipolar_distance(torch.from_numpy(m_keypoints1), torch.from_numpy(m_keypoints2), E, k1, k2)
            results['epi_errs'].append(distances)

            # if config['model'] != 'VGG19':
            #     img1 = cv2.resize(img1, None, fx=0.5, fy=0.5, interpolation= cv2.INTER_LINEAR)
            #     img2 = cv2.resize(img2, None, fx=0.5, fy=0.5, interpolation= cv2.INTER_LINEAR)

            
            key = display_canvas_matches(img1, img2, i, 100, m_keypoints1, m_keypoints2, distances, 5e-4)

            ret = estimate_pose(m_keypoints1, m_keypoints2, k1, k2, opt.estimate_pose_threshold)
            
            if ret is None:
                # results['R_errs'].append(np.inf)
                # results['t_errs'].append(np.inf)
                results['inliers'].append(np.array([]).astype(np.bool))
            else:
                R, t, inliers = ret
                t_err, R_err = relative_pose_error(rotation, translation, R, t, ignore_gt_t_thr=0.0)
                results['R_errs'].append(R_err)
                results['t_errs'].append(t_err)
                results['inliers'].append(inliers)
                
                print('R_err: ', R_err, 't_err: ',t_err)
    
    
    angular_thresholds = [5, 10, 20]
    pose_errors = np.max(np.stack([results['R_errs'], results['t_errs']]), axis=0)
    aucs = error_auc(pose_errors, angular_thresholds)
    print(aucs)
    

    precs = epidist_prec(np.array(results['epi_errs'], dtype=object), [1e-4], True)
    print(precs)

    # pose_err_20_num = 0
    # pose_err_10_num = 0
    # pose_err_5_num = 0
    # pose_err_20_ave = [0,0]
    # pose_err_10_ave = [0,0]
    # pose_err_5_ave = [0,0]
    # for i in range(len(results['R_errs'])):
    #     if results['R_errs'][i] < 20 and results['t_errs'][i] < 20:
    #         pose_err_20_num += 1
    #         pose_err_20_ave[0] += results['R_errs'][i]
    #         pose_err_20_ave[1] += results['t_errs'][i]
    #     if results['R_errs'][i] < 10 and results['t_errs'][i] < 10:
    #         pose_err_10_num += 1
    #         pose_err_10_ave[0] += results['R_errs'][i]
    #         pose_err_10_ave[1] += results['t_errs'][i]
    #     if results['R_errs'][i] < 5 and results['t_errs'][i] < 5:
    #         pose_err_5_num += 1
    #         pose_err_5_ave[0] += results['R_errs'][i]
    #         pose_err_5_ave[1] += results['t_errs'][i]


    # print('20: ',pose_err_20_num/len(results['R_errs']))

    # print('10: ',pose_err_10_num/len(results['R_errs']))

    # print('5: ',pose_err_5_num/len(results['R_errs']))

    # print('20_err_average: ',pose_err_20_ave[0]/len(results['R_errs']), ' ', pose_err_20_ave[1]/len(results['R_errs']))
    # print('10_err_average: ',pose_err_10_ave[0]/len(results['R_errs']), ' ', pose_err_10_ave[1]/len(results['R_errs']))
    # print('5_err_average: ',pose_err_5_ave[0]/len(results['R_errs']), ' ', pose_err_5_ave[1]/len(results['R_errs']))


