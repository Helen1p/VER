#coding:utf-8
import cv2
import numpy as np
import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms

from mediapipe.python.solutions import pose as mp_pose

from transformers import AutoImageProcessor, SwinModel

import tqdm

class VideoRead:
    def __init__(self, video_path, num_frames):
        self.video_path = video_path
        self.frame_length = 0
        self.num_frames = num_frames

    def get_frame(self):
        cap = cv2.VideoCapture(self.video_path)
        assert cap.isOpened()
        frames = []

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
        cap.release()
        self.frame_length = len(frames)
        return frames

    def crop_frame(self):
        """to crop frames to tensor
        return: tensor [64, 3, 224, 224]
        """
        frames = self.get_frame()  # frames: the all frames of video
        frames_tensor = []
        if self.num_frames <= len(frames):
            for i in range(self.num_frames):
                #  select 64 frames from total original frames, proportionally
                frame = frames[i * len(frames) // self.num_frames]
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = cv2.resize(frame, (224, 224))  # [3, 224, 224]
                # frame = transforms.ToTensor()(frame)
                frames_tensor.append(frame)

        else:  # if raw frames number lower than 64, padding it. 
            for i in range(self.frame_length):
                frame = frames[i]
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = cv2.resize(frame, (224, 224))  # [ 3, 224, 224]
                # frame = transforms.ToTensor()(frame)
                frames_tensor.append(frame)
            for i in range(self.num_frames - self.frame_length):
                frame = frames[self.frame_length - 1]
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = cv2.resize(frame, (224, 224))  # [ 3, 224, 224]
                # frame = transforms.ToTensor()(frame)
                frames_tensor.append(frame)
        # Frame_Tensor=torch.as_tensor(np.stack(frames_tensor))   
        Frame_Tensor = np.stack(np.stack(frames_tensor))
        return Frame_Tensor


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

image_processor = AutoImageProcessor.from_pretrained("microsoft/swin-tiny-patch4-window7-224")
model = SwinModel.from_pretrained("microsoft/swin-tiny-patch4-window7-224").to(device)

def extract_frame(root_path='./data/'):
    visual_save_base_path = os.path.join(root_path, "visual_npy")
    skeleton_save_base_path = os.path.join(root_path, "ske_npy")
    if not os.path.exists(visual_save_base_path):
        os.makedirs(visual_save_base_path)
    if not os.path.exists(skeleton_save_base_path):
        os.makedirs(skeleton_save_base_path)
    video_base_path = os.path.join(root_path, "ve8_mp4")
    wrong_video_list = []

    pose_tracker = mp_pose.Pose()

    num_frames = 100
    for file_name in tqdm.tqdm(os.listdir(video_base_path), desc='Extract visual wav'):
        try:
            video_path = os.path.join(video_base_path, file_name)
            skeleton_save_path = os.path.join(skeleton_save_base_path, file_name.replace('.mp4', '.npy'))
            visual_save_path = os.path.join(visual_save_base_path, file_name.replace('.mp4', '.npy'))
            video_rd = VideoRead(video_path, num_frames=num_frames)
            video_tensor = video_rd.crop_frame()
            video_frame_length = video_rd.frame_length
            # video_tensor [100, 224, 224, 3]
            crops = [video_tensor[i:i + 1, :, :, :] for i in range(0, num_frames)]
            ske_slice = []
            vfeat_slice = []
            with torch.no_grad():
                for crop in crops:

                    inputs = image_processor(crop.squeeze(), return_tensors="pt")

                    outputs = model(inputs['pixel_values'].cuda())

                    # visual feature
                    vfeat = outputs.last_hidden_state.mean(dim=1).detach().cpu()   # [1, 768]
                    vfeat_slice.append(vfeat)

                    result = pose_tracker.process(image=crop.squeeze())
                    pose_landmarks = result.pose_landmarks
                    # Save landmarks if pose was detected.
                    if pose_landmarks is not None:
                        # Get landmarks.
                        frame_height, frame_width = crop.squeeze().shape[0], crop.squeeze().shape[1]
                        pose_landmarks = np.array(
                            [[lmk.x * frame_width, lmk.y * frame_height, lmk.z * frame_width]
                                for lmk in pose_landmarks.landmark],
                            dtype=np.float32)
                        assert pose_landmarks.shape == (33, 3), 'Unexpected landmarks shape: {}'.format(pose_landmarks.shape)
                    else:
                        pose_landmarks = np.zeros([33, 3], dtype=float)
                    ske_slice.append(transforms.ToTensor()(pose_landmarks))
            ske_feature = torch.cat(ske_slice, dim=0)  # -> [100, 33, 3]
            v_feature = torch.cat(vfeat_slice, dim=0)  # -> [100, 768]
            np.save(skeleton_save_path, ske_feature)
            np.save(visual_save_path, v_feature)

        except Exception as e:
            print(f"Error {e}")
            wrong_video_list.append(file_name)
    print('wrong_list: ', wrong_video_list)
    print('#wrong_list: ', len(wrong_video_list))

if __name__ == '__main__':
    extract_frame()