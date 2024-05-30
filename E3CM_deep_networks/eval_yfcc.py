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

        
def to_homogeneous(points):
    return torch.cat([points, torch.ones_like(points[:, :1])],dim=1)


def compute_pose(points_A_image, points_B_image, layer, points_A, points_B, K0, K1, thresh):

    kpts0 = points_A.T
    kpts1 = points_B.T

    kpts0_image = (kpts0+0.5)*(2**layer)+0.5
    kpts1_image = (kpts1+0.5)*(2**layer)+0.5

    kpts_confidence = np.arange(kpts0.shape[0])

    for i in range(kpts0.shape[0]):
        points_A_image_dis = np.sum(np.square(points_A_image - kpts0_image[i,:].reshape(2,1)),axis=0)
        confidence = (np.where(points_A_image_dis<((2**(layer-1)+1)**2)*2))[0].shape[0]
        kpts_confidence[i] = confidence
    
    chosen_index = np.argsort(-kpts_confidence)[:8]
    kpts0_chosen = kpts0[chosen_index,:]
    kpts1_chosen = kpts1[chosen_index,:]


    kpts0_chosen = (kpts0 - K0[[0, 1], [2, 2]][None]) / K0[[0, 1], [0, 1]][None]
    kpts1_chosen = (kpts1 - K1[[0, 1], [2, 2]][None]) / K1[[0, 1], [0, 1]][None]

    # normalize ransac threshold
    ransac_thr = thresh / np.mean([K0[0, 0], K0[1, 1], K1[0, 0], K1[1, 1]])

    # compute pose with cv2
    E, mask = cv2.findEssentialMat(
        kpts0_chosen, kpts1_chosen, np.eye(3), method=cv2.FM_8POINT)

    return E

def outlier_rejection(points_A, points_B, threshold, E, k1, k2):

    if E.shape[0] > 3:
        E = E[:3,:]

    points_A_refined = []
    points_B_refined = []

    distances = symmetric_epipolar_distance(torch.from_numpy(points_A.T), torch.from_numpy(points_B.T), E, k1, k2)

    for i in range(points_A.T.shape[0]):
        if distances[i] < threshold:
            points_A_refined.append(torch.from_numpy(points_A.T)[i,:])
            points_B_refined.append(torch.from_numpy(points_B.T)[i,:])
    
    if len(points_A_refined) == 0:
        return points_A, points_B
  
    points_A_refined = torch.stack(points_A_refined) 
    points_B_refined = torch.stack(points_B_refined) 

    return points_A_refined.t().numpy(), points_B_refined.t().numpy()

# def outlier_rejection_confidence(points_A, points_B, confidence_threshold, Fundmat):

#     if Fundmat.shape[0] == 9:
#         Fundmat = Fundmat[:3,:]

#     keypoints_1 = points_A.T
#     keypoints_2 = points_B.T

#     keypoints_num = keypoints_1.shape[0]

#     residules = (to_homogeneous(torch.from_numpy(keypoints_1)).numpy() @ Fundmat @ to_homogeneous(torch.from_numpy(keypoints_2)).numpy().T).diagonal()
    
#     rank_index = np.argsort(residules)

#     confidences = 13 * np.log10(abs(rank_index / (keypoints_num * residules)))
#     # rank = np.argsort(confidences)
#     # confidences_rank = confidences[rank]
#     # print(confidences_rank[int(keypoints_num*0.3)])
#     satisfied_keypoints_index = np.where(confidences > confidence_threshold)
        
#     refined_keypoints_1 = keypoints_1[satisfied_keypoints_index]
#     refined_keypoints_2 = keypoints_2[satisfied_keypoints_index]
    
#     return refined_keypoints_1.T, refined_keypoints_2.T

def refinement_pose_ransac(points_A_list, points_B_list, outlier_rejection_threshold, estimate_pose_thresh, k1, k2):

    k1 /= 8
    k1[2][2] = 1.0
    k2 /= 8
    k2[2][2] = 1.0

    if points_A_list[0].shape[1] < 8:
        points_A = points_A_list[3]
        points_B = points_B_list[3]
    else :
        E = compute_pose(points_A_list[3], points_B_list[3], 3, points_A_list[0], points_B_list[0], k1, k2, estimate_pose_thresh)

        for i in range(1,4):
            k1 *= 2
            k1[2][2] = 1.0
            k2 *= 2
            k2[2][2] = 1.0
            if type(E) == type(None):
                points_A = points_A_list[3]
                points_B = points_B_list[3]
            else:
                points_A = points_A_list[i] #shape: 2 * num
                points_B = points_B_list[i]
                if points_A.shape[1] < 8:
                    points_A = points_A_list[3]
                    points_B = points_B_list[3]
                else :
                    points_A, points_B = outlier_rejection(points_A, points_B, outlier_rejection_threshold, E, k1, k2)      
                    E = compute_pose(points_A_list[3], points_B_list[3], (3-i), points_A, points_B, k1, k2, estimate_pose_thresh)
   
    return points_A, points_B

def display_canvas_matches(img1, img2, num, wait_time, m_pts1, m_pts2, distances, threshold):
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

        
        if distances[i] < threshold:
            color = (0, 255, 0)
        if distances[i] >= threshold:
            color = (0, 0, 255)
        cv2.line(canvas, kp1, kp2, color, 2)

    cv2.imshow("canvas", canvas)
    key = cv2.waitKey(wait_time)
    cv2.imwrite('canvas_'+str(num)+'.jpg', canvas)
    
    return key 


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


def data_from_txt(data_path):

    data = {
        'img1_path': [],
        'img2_path': [],
        'intrinsics1': [],
        'intrinsics2': [],
        'rotation':[],
        'translation':[]
    }

    

    with open(data_path, "r") as f:
        
        for line in f.readlines():

            intrinsics1 = np.eye(3)
            intrinsics2 = np.eye(3)
            rotation = np.eye(3)
            translation = np.array([1.0,1.0,1.0])

            line_element = line.split()

            data['img1_path'].append(line_element[0])
            data['img2_path'].append(line_element[1])
            
            intrinsics1[0][0] = line_element[4]
            intrinsics1[0][1] = line_element[5]
            intrinsics1[0][2] = line_element[6]
            intrinsics1[1][0] = line_element[7]
            intrinsics1[1][1] = line_element[8]
            intrinsics1[1][2] = line_element[9]
            intrinsics1[2][0] = line_element[10]
            intrinsics1[2][1] = line_element[11]
            intrinsics1[2][2] = line_element[12]

            data['intrinsics1'].append(intrinsics1)

            intrinsics2[0][0] = line_element[13]
            intrinsics2[0][1] = line_element[14]
            intrinsics2[0][2] = line_element[15]
            intrinsics2[1][0] = line_element[16]
            intrinsics2[1][1] = line_element[17]
            intrinsics2[1][2] = line_element[18]
            intrinsics2[2][0] = line_element[19]
            intrinsics2[2][1] = line_element[20]
            intrinsics2[2][2] = line_element[21]

            data['intrinsics2'].append(intrinsics2)

            rotation[0][0] = line_element[22]
            rotation[0][1] = line_element[23]
            rotation[0][2] = line_element[24]
            rotation[1][0] = line_element[26]
            rotation[1][1] = line_element[27]
            rotation[1][2] = line_element[28]
            rotation[2][0] = line_element[30]
            rotation[2][1] = line_element[31]
            rotation[2][2] = line_element[32]

            data['rotation'].append(rotation)

            translation[0] = line_element[25]
            translation[1] = line_element[29]
            translation[2] = line_element[33]

            data['translation'].append(translation)

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
        default='/media/zcb/Data/OANet/raw_data/yfcc100m',
        help='Path to dataset')
    parser.add_argument('--estimate_pose_threshold', type=float, default=1,
        help='threshold in estimating pose')
    parser.add_argument('--outlier_rejection_threshold', type=float, default=1e-4,
        help='outlier_rejection_threshold')
    parser.add_argument('--refinement_pose', type=bool, default=True,
        help='if refinement_pose')
    opt = parser.parse_args()
    
    # from DFM_matcher import DeepFeatureMatcher

    # with open("config.yml", "r") as configfile:
    #     config = yaml.safe_load(configfile)['configuration']
    # fm = DeepFeatureMatcher(enable_two_stage = config['enable_two_stage'], model = config['model'], 
    #                     ratio_th = config['ratio_th'], bidirectional = config['bidirectional'], )
    
    data = data_from_txt(opt.data_root + '/yfcc_test_pairs_with_gt.txt')

    results={'R_errs':[],
              't_errs':[],
              'inliers':[],
              'epi_errs':[]}

    sift = cv2.SIFT_create()

    with torch.no_grad():

        loss_sum = 0
        for i in range(len(data['img1_path'])):
            #len(data['img1_path'])
            img1_path = opt.data_root + '/' + data['img1_path'][i]
            img2_path = opt.data_root + '/' + data['img2_path'][i]

            
            if (os.path.isfile(img1_path) and os.path.isfile(img2_path)):

                img1 = cv2.imread(img1_path)
                img2 = cv2.imread(img2_path)

                # fx_1 = 1200/max(img1.shape[0], img1.shape[1])
                # fx_2 = 1200/max(img2.shape[0], img2.shape[1])

                # img1 = cv2.resize(img1, (0,0), fx=fx_1, fy=fx_1, interpolation=cv2.INTER_LINEAR)
                # img2 = cv2.resize(img2, (0,0), fx=fx_2, fy=fx_2, interpolation=cv2.INTER_LINEAR)

                intrinsics1 = data['intrinsics1'][i]
                intrinsics2 = data['intrinsics2'][i]

                # intrinsics1 *= fx_1
                # intrinsics1[2][2] = 1.0
                # intrinsics2 *= fx_2
                # intrinsics2[2][2] = 1.0

                k1 = intrinsics1
                k2 = intrinsics2

                rotation = data['rotation'][i]
                translation = data['translation'][i]

                E = translation_mat(translation) @ rotation

                # if config['model'] != 'VGG19':
                #     # img1 = cv2.pyrUp(img1)
                #     # img2 = cv2.pyrUp(img2)
                #     img1 = cv2.resize(img1, None, fx=2, fy=2, interpolation= cv2.INTER_LINEAR)
                #     img2 = cv2.resize(img2, None, fx=2, fy=2, interpolation= cv2.INTER_LINEAR)
                

                # _, _, points_A_list, points_B_list = fm.match_small_img(img1, img2)

                # if opt.refinement_pose:
                #     points_A, points_B = refinement_pose_ransac(points_A_list, points_B_list, opt.outlier_rejection_threshold, opt.estimate_pose_threshold, k1, k2)
                
                # else:
                #     points_A = points_A_list[3]
                #     points_B = points_B_list[3]

                # m_keypoints1 = points_A.T
                # m_keypoints2 = points_B.T

                kp1, des1 = sift.detectAndCompute(img1, None)
                kp2, des2 = sift.detectAndCompute(img2, None)

                bf = cv2.BFMatcher_create(cv2.NORM_L1, crossCheck=False)
                matches = bf.match(des1, des2)

                m_keypoints1 = np.ones((len(matches),2))
                m_keypoints2 = np.ones((len(matches),2))

                for k in range(len(matches)):
                    m_keypoints1[k][0] = kp1[matches[k].queryIdx].pt[0]
                    m_keypoints1[k][1] = kp1[matches[k].queryIdx].pt[1]
                    m_keypoints2[k][0] = kp2[matches[k].trainIdx].pt[0]
                    m_keypoints2[k][1] = kp2[matches[k].trainIdx].pt[1]

                distances = symmetric_epipolar_distance(torch.from_numpy(m_keypoints1), torch.from_numpy(m_keypoints2), E, k1, k2)
                results['epi_errs'].append(distances)

                m_keypoints1_homo = to_homogeneous(torch.from_numpy(m_keypoints1)).numpy()
                m_keypoints2_homo = to_homogeneous(torch.from_numpy(m_keypoints2)).numpy()

                loss_matrix = (np.linalg.inv(k2) @ m_keypoints1_homo.T).T @ translation_mat(translation) @ rotation @ (np.linalg.inv(k1) @ m_keypoints2_homo.T)
                loss = np.mean(abs(np.diagonal(loss_matrix)))
                loss_sum += loss

                # key = display_canvas_matches(img1, img2, i, 100, m_keypoints1, m_keypoints2, distances, 5e-2)

                ret = estimate_pose(m_keypoints1, m_keypoints2, intrinsics1, intrinsics2, opt.estimate_pose_threshold)
                        
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
            
            else:
                print('not found')
        
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


