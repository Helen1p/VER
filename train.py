# coding:utf-8

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
from models.TransVER import NERModel

from datasets.VE8_loader import VE8Dataset
from torch.utils.data import DataLoader

train_set = VE8Dataset()
train_loader = DataLoader(train_set, batch_size=16, pin_memory=False, shuffle=True, num_workers=8)

# 没有valid集
# val_set = VE8Dataset()
# val_loader = DataLoader(val_set, batch_size=16, pin_memory=False, shuffle=True, num_workers=8)

test_set = VE8Dataset()
test_loader = DataLoader(test_set, batch_size=16, pin_memory=False, shuffle=True, num_workers=8)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = NERModel()
model.to(device)

# Optimizer and criterion

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training

EPOCHS = 100

for epoch in range(EPOCHS):

    model.train()
    running_loss = 0.0

    for batch_idx, data in enumerate(train_loader):
        # get inputs and labels
        feat_v, feat_a, feat_s, labels = data
        feat_v = feat_v.to(device)
        feat_a = feat_a.to(device)
        feat_s = feat_s.to(device)
        labels = labels.to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = model(feat_v, feat_a, feat_s)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()

        if (batch_idx + 1) % 1000 == 0:
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                  .format(epoch + 1, EPOCHS, batch_idx + 1, len(train_loader), loss.item()))

    ckpt_name = 'ckpt' + str(epoch+1)
    if not os.path.exists('./checkpoint/{0}/'.format(ckpt_name)):
        os.mkdir('./checkpoint/{0}/'.format(ckpt_name))
    checkpoint = {
        'epoch': epoch,
        'state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        # 'trainLosses': trainLosses,
    }
    torch.save(checkpoint, './checkpoint/{0}/'.format(ckpt_name) + str(epoch+1) + '.pt')


    # evaluate on val set
    # model.eval()
    #
    # for batch_idx, data in enumerate(val_loader):
    #     # get inputs and labels
    #     feat_v, feat_a, feat_s, labels = data
    #     feat_v = feat_v.to(device)
    #     feat_a = feat_a.to(device)
    #     feat_s = feat_s.to(device)
    #     labels = labels.to(device)
    #
    #     # zero the parameter gradients
    #     optimizer.zero_grad()
    #
    #     # forward + backward + optimize
    #     outputs = model(feat_v, feat_a, feat_s)
    #     loss = criterion(outputs, labels)
    #     loss.backward()
    #     optimizer.step()
    #
    #     # print statistics
    #     running_loss += loss.item()
    #
    #     if (batch_idx + 1) % 1000 == 0:
    #         print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
    #               .format(epoch + 1, EPOCHS, batch_idx + 1, len(train_loader), loss.item()))
