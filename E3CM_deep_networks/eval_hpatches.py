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

#   tool-function

# rsz_img = cv.resize(img, (0, 0), fx=0.25, fy=0.25, interpolation=cv.INTER_NEAREST)

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



class HpatchesDataLoader():
    def __init__(self, dataset_path, img_format, gray=True):
        self.pairs = []
        dirlist = []
        self.gray = gray
        for root, dirnames, filenames in os.walk(dataset_path):
            dirnames = sorted(dirnames)
            for dirname in dirnames:
                dirlist.append(os.path.join(root, dirname))
                data_path = os.path.join(root, dirname)
            #   traverse images
                image_pairs = glob.glob(data_path + "/*" + img_format)
                image_pairs = sorted(image_pairs)
                image_num = len(image_pairs)
                for j in range(1,image_num):
                    id2 = j + 1
                    real_H = get_gt_H_matrix(data_path, 1, id2)
                    self.pairs.append([image_pairs[0], image_pairs[j], 1, id2, real_H])
        self.read_idx = 0
        self.read_pair = []
        self.length = len(self.pairs)

    def get_length(self):
        return self.length

    def next_item(self):
        self.read_pair = self.pairs[self.read_idx]
        self.read_idx = self.read_idx + 1

    def read_data(self):
        self.next_item()
        img1 = cv2.imread(self.read_pair[0], 0)
        img2 = cv2.imread(self.read_pair[1], 0)
        id1 = self.read_pair[2]
        id2 = self.read_pair[3]
        real_H = self.read_pair[4]
        return img1, img2, id1, id2, real_H

    def read_data_from_index(self, index : int):
        if self.gray:
            img1 = cv2.imread(self.pairs[index][0], 0)
            img2 = cv2.imread(self.pairs[index][1], 0)
            id1 = self.pairs[index][2]
            id2 = self.pairs[index][3]
            real_H = self.pairs[index][4]
            return img1, img2, id1, id2, real_H
        else:
            img1 = cv2.imread(self.pairs[index][0])
            img2 = cv2.imread(self.pairs[index][1])
            id1 = self.pairs[index][2]
            id2 = self.pairs[index][3]
            real_H = self.pairs[index][4]
            return img1, img2, id1, id2, real_H

#   color bar for display confidence(heat) valueimg_test2
gray_bar = np.zeros((1, 256), np.uint8)
for i in range(256):
    gray_bar[0][i] = i
color_bar = cv2.applyColorMap(gray_bar, cv2.COLORMAP_JET)



def read_data_from_hpatches(data_path, img1_num, img2_num):

    return img1, img2

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


    # from dfm_se import DeepFeatureMatcher
    from DFM_matcher import DeepFeatureMatcher

    with open("config.yml", "r") as configfile:
        config = yaml.safe_load(configfile)['configuration']
    fm = DeepFeatureMatcher(enable_two_stage = config['enable_two_stage'], model = config['model'],
                        ratio_th = config['ratio_th'], bidirectional = config['bidirectional'], )
    
    img_list = []
    homo_list = []

    for root, dirnames, filenames in os.walk(opt.dataset_path):
        for name in dirnames:
            img_list.append(os.path.join(root, name))
        for name in filenames:
            homo_list.append(os.path.join(root, name))


    with torch.no_grad():
        results = {'illusion':[],
                   'viewpoint':[],
                   'overall':[]}
        for thres in range(1, 2):
            count = 0
            value_sum1 = 0.0
            value_sum2 = 0.0
            value_sum3 = 0.0
            value_sum4 = 0.0
            value_sum5 = 0.0
            value_sum6 = 0.0
            value_sum7 = 0.0
            value_sum8 = 0.0
            value_sum9 = 0.0
            value_sum10 = 0.0
            matches_num = 0
            for root in img_list:          
                for match in range(2,7):
                    # img1 = cv2.imread(root + '/1.ppm')
                    # img2 = cv2.imread(root + '/' + str(match) + '.ppm')

                    img1 = cv2.imread('/media/zcb/Windows-SSD/hpatches-sequences-release/v_maskedman/1.ppm')
                    img2 = cv2.imread('/media/zcb/Windows-SSD/hpatches-sequences-release/v_maskedman/3.ppm')
                    
                    img_H, img_W, _ = img1.shape

                    if config['model'] != 'VGG19':
                        # img1 = cv2.pyrUp(img1)
                        # img2 = cv2.pyrUp(img2)
                        img1 = cv2.resize(img1, None, fx=2, fy=2, interpolation= cv2.INTER_LINEAR)
                        img2 = cv2.resize(img2, None, fx=2, fy=2, interpolation= cv2.INTER_LINEAR)

                    

                    real_H = get_H_matrix(root + '/H_1_' + str(match))
                    # img1, img2, id1, id2, real_H = rhdl.read_data_from_index(index)
                    #print(type(img1))                  

                    count = count + 1

                    _, _, points_A_list, points_B_list = fm.match_large_img(img1, img2)

                    #preprocess keypoints_lists

                    # for i in range(0,4,1):
                    #     points_lists_confidence = np.arange(points_A_list[i].shape[1])

                    #     for k in range(points_A_list[i].shape[1]):
                    #         dif = points_A_list[4]-points_A_list[i][:,k].reshape(2,1)
                    #         _, index = np.where(abs(dif) < 2**(3-i))
                    #         confi = index.shape[0] - np.unique(index).shape[0]
                    #         points_lists_confidence[k] = confi
                        
                    #     points_A_list[i] = points_A_list[i][:,np.where(points_lists_confidence >= 2**(3-i))].reshape(2,-1)
                    #     points_B_list[i] = points_B_list[i][:,np.where(points_lists_confidence >= 2**(3-i))].reshape(2,-1)

                    # hierarchical pose ransac
                    if opt.refinement_pose:
                        points_A, points_B = refinement_pose_ransac(points_A_list, points_B_list, opt.confidence_threshold)
                    
                    else:
                        points_A = points_A_list[4]
                        points_B = points_B_list[4]

                    matches_num += points_A.shape[1]

                    m_keypoints1 = points_A.T
                    m_keypoints2 = points_B.T
                    
                    key = display_canvas_matches(img1, img2, match, 100, m_keypoints1, m_keypoints2)
                    # m_confs = matches["match_score"]
                    mma1 = get_MMA(real_H, m_keypoints1, m_keypoints2, 1)
                    mma2 = get_MMA(real_H, m_keypoints1, m_keypoints2, 2)
                    mma3 = get_MMA(real_H, m_keypoints1, m_keypoints2, 3)
                    mma4 = get_MMA(real_H, m_keypoints1, m_keypoints2, 4)
                    mma5 = get_MMA(real_H, m_keypoints1, m_keypoints2, 5)
                    mma6 = get_MMA(real_H, m_keypoints1, m_keypoints2, 6)
                    mma7 = get_MMA(real_H, m_keypoints1, m_keypoints2, 7)
                    mma8 = get_MMA(real_H, m_keypoints1, m_keypoints2, 8)
                    mma9 = get_MMA(real_H, m_keypoints1, m_keypoints2, 9)
                    mma10 = get_MMA(real_H, m_keypoints1, m_keypoints2, 10)
                    value_sum1 = value_sum1 + mma1
                    value_sum2 = value_sum2 + mma2
                    value_sum3 = value_sum3 + mma3
                    value_sum4 = value_sum4 + mma4
                    value_sum5 = value_sum5 + mma5
                    value_sum6 = value_sum6 + mma6
                    value_sum7 = value_sum7 + mma7
                    value_sum8 = value_sum8 + mma8
                    value_sum9 = value_sum9 + mma9
                    value_sum10 = value_sum10 + mma10
                #   print MMA info
                    # if (count % 10) == 0:
                    if 1:
                        print("count : %4d" % count, "thres : %4d" % 1, "   Average MMA : %.6f" % (value_sum1/count))
                        print("count : %4d" % count, "thres : %4d" % 2, "   Average MMA : %.6f" % (value_sum2/count))
                        print("count : %4d" % count, "thres : %4d" % 3, "   Average MMA : %.6f" % (value_sum3/count))
                        print("count : %4d" % count, "thres : %4d" % 4, "   Average MMA : %.6f" % (value_sum4/count))
                        print("count : %4d" % count, "thres : %4d" % 5, "   Average MMA : %.6f" % (value_sum5/count))
                        print("count : %4d" % count, "thres : %4d" % 6, "   Average MMA : %.6f" % (value_sum6/count))
                        print("count : %4d" % count, "thres : %4d" % 7, "   Average MMA : %.6f" % (value_sum7/count))
                        print("count : %4d" % count, "thres : %4d" % 8, "   Average MMA : %.6f" % (value_sum8/count))
                        print("count : %4d" % count, "thres : %4d" % 9, "   Average MMA : %.6f" % (value_sum9/count))
                        print("count : %4d" % count, "thres : %4d" % 10, "   Average MMA : %.6f" % (value_sum10/count))
                    
                    if count == 330:
                        results['illusion'].append(value_sum1/count)
                        results['illusion'].append(value_sum2/count)
                        results['illusion'].append(value_sum3/count)
                        results['illusion'].append(value_sum4/count)
                        results['illusion'].append(value_sum5/count)
                        results['illusion'].append(value_sum6/count)
                        results['illusion'].append(value_sum7/count)
                        results['illusion'].append(value_sum8/count)
                        results['illusion'].append(value_sum9/count)
                        results['illusion'].append(value_sum10/count)
                    
                    if count == 575:
                        results['overall'].append(value_sum1/count)
                        results['overall'].append(value_sum2/count)
                        results['overall'].append(value_sum3/count)
                        results['overall'].append(value_sum4/count)
                        results['overall'].append(value_sum5/count)
                        results['overall'].append(value_sum6/count)
                        results['overall'].append(value_sum7/count)
                        results['overall'].append(value_sum8/count)
                        results['overall'].append(value_sum9/count)
                        results['overall'].append(value_sum10/count)

                        results['viewpoint'].append((results['overall'][0]*575-results['illusion'][0]*330)/245)
                        results['viewpoint'].append((results['overall'][1]*575-results['illusion'][1]*330)/245)
                        results['viewpoint'].append((results['overall'][2]*575-results['illusion'][2]*330)/245)
                        results['viewpoint'].append((results['overall'][3]*575-results['illusion'][3]*330)/245)
                        results['viewpoint'].append((results['overall'][4]*575-results['illusion'][4]*330)/245)
                        results['viewpoint'].append((results['overall'][5]*575-results['illusion'][5]*330)/245)
                        results['viewpoint'].append((results['overall'][6]*575-results['illusion'][6]*330)/245)
                        results['viewpoint'].append((results['overall'][7]*575-results['illusion'][7]*330)/245)
                        results['viewpoint'].append((results['overall'][8]*575-results['illusion'][8]*330)/245)
                        results['viewpoint'].append((results['overall'][9]*575-results['illusion'][9]*330)/245)

            print('average matches: ',matches_num/count)    
            print(results)
