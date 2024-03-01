
### 数据存放
./data/ve8_mp4/ 存放所有视频的mp4


./data/audio_wav/ 存放所有视频的wav
./data/audio_log_mel/ 存放所有视频的预处理特征
./data/audio_npy/ 存放所有视频的 音频特征
./data/ske_npy/ 存放所有视频的 骨架点特征
./data/visual_npy/ 存放所有视频的 视频特征


#### 提取音频特征
CUDA_VISIBLE_DEVICES=1 python tools/extract_feature_a.py 

#### 提取视频特征 + 骨架点特征
CUDA_VISIBLE_DEVICES=1 python tools/extract_feature_v.py 



### VideoEmotion-8
* Download the videos [here](https://drive.google.com/drive/folders/0B5peJ1MHnIWGd3pFbzMyTG5BSGs?resourcekey=0-hZ1jo5t1hIauRpYhYIvWYA&usp=sharing).
* Convert from mp4 to mp3 files using ```/tools/video2mp3.py```
* 

### VideoEmotion-8
* Download the videos [here](https://drive.google.com/drive/folders/0B5peJ1MHnIWGd3pFbzMyTG5BSGs?resourcekey=0-hZ1jo5t1hIauRpYhYIvWYA&usp=sharing).
* Convert from mp4 to jpg files using ```/tools/video2jpg.py```
* Add n_frames information using ```/tools/n_frames.py```
* Generate annotation file in json format using ```/tools/ve8_json.py```
* Convert from mp4 to mp3 files using ```/tools/video2mp3.py```

## Running the code
Assume the strcture of data directories is the following:
```misc
~/
  VideoEmotion8--imgs
    .../ (directories of class names)
      .../ (directories of video names)
        .../ (jpg files)
  VideoEmotion8--mp3
    .../ (directories of class names)
      .../ (mp3 files)
  results
  ve8_01.json
```

Confirm all options in ```~/opts.py```.
```bash
python main.py
```
