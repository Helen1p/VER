# coding:utf-8

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
from models.TransVER import NERModel
from datasets.VE8_loader import VE8Dataset
from torch.utils.data import DataLoader

test_set = VE8Dataset()
test_loader = DataLoader(test_set, batch_size=16, pin_memory=False, shuffle=True, num_workers=8)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = NERModel()
model.to(device)

lastckpt = '/root/autodl-tmp/checkpoint/ckpt2/2.pt'

checkpoint = torch.load(lastckpt)
model.load_state_dict(checkpoint['state_dict'], strict=False)

# Testing
EPOCHS = 1
with torch.no_grad():
    right_all = 0
    number = 0
    for batch_idx, data in enumerate(test_loader):
        # get inputs and labels
        feat_v, feat_a, feat_s, labels = data
        feat_v = feat_v.to(device)
        feat_a = feat_a.to(device)
        feat_s = feat_s.to(device)
        labels = labels.to(device)

        outputs = model(feat_v, feat_a, feat_s)
        pre = torch.argmax(outputs,dim=-1)+1
        print('predict: ', pre, ' gd: ', labels)
        right = sum(pre==labels)
        number += len(labels)
        right_all += right
    print('Accuracy: ', right_all/number)



