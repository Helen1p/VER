#coding:utf-8

import os
import numpy as np
from torch.utils.data import Dataset, DataLoader
import json


def read_numpy(file_path):
    return np.load(file_path)

# class VE8Dataset(Dataset):
#
#     def __init__(self, root_path='./data/'):
#         """
#         :param root_path: root path
#         """
#         self.root_path = root_path
#         self.video_path = os.path.join(self.root_path, 've8_mp4')
#         self.vfeat_path = os.path.join(self.root_path, 'visual_npy')  # train or valid
#         self.afeat_path = os.path.join(self.root_path, 'audio_npy')
#         self.sfeat_path = os.path.join(self.root_path, 'ske_npy')
#         # label
#         self.video_dir = os.listdir(self.video_path)
#         self.label_list = ["Anger", "Anticipation", "Disgust", "Fear", "Joy", "Sadness", "Surprise", "Trust"]
#
#
#     def __getitem__(self, inx):
#         """ get data item
#         :param  video_tensor, label
#         """
#         video_file_name = self.video_dir[inx]
#         v_feat = read_numpy(os.path.join(self.vfeat_path, video_file_name.replace('.mp4', '.ny')))
#         a_feat = read_numpy(os.path.join(self.afeat_path, video_file_name.replace('.mp4', '.npy')))
#         s_feat = read_numpy(os.path.join(self.sfeat_path, video_file_name.replace('.mp4', '.npy')))
#         # label
#         with open("./tool/annotations/ve8/ve8_01.json", "r", encoding="utf-8") as f:
#             content = json.load(f)
#             name = os.path.splitext(video_file_name)[0]
#             label = self.label_list.index(content['database'][name]['annotations']['label']) + 1
#         return v_feat, a_feat, s_feat, label
#
#     def __len__(self):
#         """:return the number of video """
#         return len(self.video_dir)


class VE8Dataset(Dataset):

    def __init__(self, root_path='./data/'):
        """
        :param root_path: root path
        """
        self.root_path = root_path
        self.video_path = os.path.join(self.root_path, 've8_mp4')
        self.vfeat_path = os.path.join(self.root_path, 'visual_npy')  # train or valid
        self.afeat_path = os.path.join(self.root_path, 'audio_npy')
        self.sfeat_path = os.path.join(self.root_path, 'ske_npy')
        # label
        self.video_dir = os.listdir(self.video_path)
        self.label_list = ["Anger", "Anticipation", "Disgust", "Fear", "Joy", "Sadness", "Surprise", "Trust"]


    def __getitem__(self, inx):
        """ get data item
        :param  video_tensor, label
        """
        video_file_name = self.video_dir[inx]
        v_feat = read_numpy(os.path.join(self.vfeat_path, video_file_name.replace('.mp4', '.ny')))
        a_feat = read_numpy(os.path.join(self.afeat_path, video_file_name.replace('.mp4', '.npy')))
        s_feat = read_numpy(os.path.join(self.sfeat_path, video_file_name.replace('.mp4', '.npy')))
        # label
        with open("./tool/annotations/ve8/ve8_01.json", "r", encoding="utf-8") as f:
            content = json.load(f)
            name = os.path.splitext(video_file_name)[0]
            label = self.label_list.index(content['database'][name]['annotations']['label']) + 1
        return v_feat, a_feat, s_feat, label

    def __len__(self):
        """:return the number of video """
        return len(self.video_dir)

