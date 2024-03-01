#coding:utf-8
import os

import torch
import librosa
from moviepy.editor import VideoFileClip
import tqdm
# from torchvggish import vggish, vggish_input
from torchvggish_gpu import vggish
from torchvggish import vggish_input
import numpy as np
import pickle


def load_pkl(file_path):
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
        f.close()
    return data
def extract_one_log_mel(wav_path, lm_save_path):
    """extract the .wav file from one video and save to .pkl"""
    log_mel_tensor = vggish_input.wavfile_to_examples(wav_path)

    N_SECONDS, CHANNEL, N_BINS, N_BANDS = log_mel_tensor.shape
    new_lm_tensor = torch.zeros(100, CHANNEL, N_BINS, N_BANDS)
    if N_SECONDS <=100:
        new_lm_tensor[:N_SECONDS] = log_mel_tensor
        # new_lm_tensor[N_SECONDS:] = log_mel_tensor[-1].repeat(100-N_SECONDS, 1, 1, 1)
    else:
        new_lm_tensor = log_mel_tensor[:100, :, :, :]
    log_mel_tensor = new_lm_tensor

    with open(lm_save_path, "wb") as fw:
        pickle.dump(log_mel_tensor, fw)

def split_audio(video_path, wav_save_path):
    """extract the .wav file from one video"""
    video = VideoFileClip(video_path)
    audio = video.audio
    audio.write_audiofile(wav_save_path, fps=16000)

def extract_audio_wav(root_path="./data/"):
    """extract the .wav files for videos"""
    wav_save_base_path = os.path.join(root_path, "audio_wav")
    if not os.path.exists(wav_save_base_path):
        os.makedirs(wav_save_base_path)
    video_base_path = os.path.join(root_path, "ve8_mp4")
    wrong_video_list = []
    count = 0
    for file_name in tqdm.tqdm(os.listdir(video_base_path), desc='Extract audio wav'):
        try:
            video_path = os.path.join(video_base_path, file_name)
            wav_save_path = os.path.join(wav_save_base_path, file_name).replace('.mp4', '.wav')
            split_audio(video_path, wav_save_path)
            count += 1
        except Exception as e:
            print(f"Error {e}")
            wrong_video_list.append(file_name)
    print('wrong_list: ', wrong_video_list)
    print('#wrong_list: ', len(wrong_video_list))

def extract_audio_log_mel(root_path="./data/"):
    """extract and save the log_mel map for each .wav"""
    wav_base_path = os.path.join(root_path, "audio_wav")
    lm_save_base_path= os.path.join(root_path, "audio_log_mel")
    if not os.path.exists(lm_save_base_path):
        os.makedirs(lm_save_base_path)
    wrong_video_list = []
    count = 0
    for file_name in tqdm.tqdm(os.listdir(wav_base_path), desc='Extract audio log mel'):
        try:
            wav_path = os.path.join(wav_base_path, file_name)
            lm_save_path = os.path.join(lm_save_base_path, file_name).replace('.wav', '.pkl')
            wrong_item = extract_one_log_mel(wav_path, lm_save_path)
            count += 1
        except Exception as e:
            print(f"Error {e}")
            wrong_video_list.append(file_name)
    print('wrong_list: ', wrong_video_list)
    print('#wrong_list: ', len(wrong_video_list))

def extract_audio_feature(root_path='./data/'):
    wav_base_path = os.path.join(root_path, "audio_wav")
    wav_pkl_base_path = os.path.join(root_path, 'audio_log_mel')
    
    npy_base_path = os.path.join(root_path, 'audio_npy')
    if not os.path.exists(npy_base_path):
        os.makedirs(npy_base_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = vggish().to(device)
    model.cuda()
    model.eval()

    wrong_video_list = []

    with torch.no_grad():
        for file_name in tqdm.tqdm(os.listdir(wav_base_path), desc='Extract audio feature'):
            try:
                wav_path = os.path.join(wav_base_path, file_name)
                wav_pkl_path = os.path.join(wav_pkl_base_path, file_name.replace('.wav', '.pkl'))
                # 从aduio
                # example = vggish_input.wavfile_to_examples(wav_path).to(device)
                # 直接使用提取好的log xx
                example = load_pkl(wav_pkl_path).to(device)
                audio_feature = model(example) # [100, 128]
                npy_path = os.path.join(npy_base_path, file_name.replace('.wav', '.npy'))
                np.save(npy_path, audio_feature.detach().cpu().numpy())
            except Exception as e:
                print(f"Error {e}")
                wrong_video_list.append(file_name)
    print(wrong_video_list)

if __name__ == "__main__":
    extract_audio_wav()
    extract_audio_log_mel()
    extract_audio_feature()

# CUDA_VISIBLE_DEVICES=0 python tools/extract_feature_a.py
