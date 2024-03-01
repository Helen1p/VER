# coding:utf-8
import torch
import torch.nn as nn
import math

from models.vlad_head import NeXtVLAD


# https://github.com/moskomule/senet.pytorch/blob/master/senet/se_module.py
class SE(nn.Module):
    def __init__(self, feat_dim=1024, gating_reduction=8, drop_rate=0.0):
        super(SE, self).__init__()
        self.drop_rate = drop_rate
        self.gating_reduction = gating_reduction  # gating_reduction

        self.drop_fusion = nn.Dropout(self.drop_rate)

        self.fc_reduce = nn.Sequential(
            nn.Linear(feat_dim, 512, bias=False),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
        )

        self.fc = nn.Sequential(
            nn.Linear(512, 512 // self.gating_reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(512 // self.gating_reduction),
            nn.Linear(512 // self.gating_reduction, 512, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        # 先进行Dropout
        x = self.drop_fusion(x)
        # 先降维
        x = self.fc_reduce(x)
        b = x.size(0)
        y = self.fc(x).view(b, -1)
        return x * y.expand_as(x)


class TransEncoder(nn.Module):
    '''standard transformer encoder'''

    def __init__(self, d_model, n_head, dim_ff, dropout=0.0, num_layers=1, num_frames=64):
        super(TransEncoder, self).__init__()
        self.pos_encoder = PositionalEncoding(d_model, 0.1, num_frames)

        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model,
                                                   nhead=n_head,
                                                   dim_feedforward=dim_ff,
                                                   dropout=dropout,
                                                   activation='relu')
        encoder_norm = nn.LayerNorm(d_model)
        self.trans_encoder = nn.TransformerEncoder(encoder_layer, num_layers, encoder_norm)

    def forward(self, src):
        src = self.pos_encoder(src)
        e_op = self.trans_encoder(src)
        return e_op


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        x = self.dropout(x)
        return x


class NERModel(nn.Module):
    def __init__(self, d_model=512, dropout=0.1):
        super(NERModel, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.d_model = d_model
        self.len_v = 100
        self.len_a = 100
        self.len_s = 100

        self.enco_v = TransEncoder(d_model=d_model, n_head=4, dropout=0.1, dim_ff=2048, num_layers=2,
                                   num_frames=self.len_v)
        self.enco_a = TransEncoder(d_model=d_model, n_head=4, dropout=0.1, dim_ff=2048, num_layers=2,
                                   num_frames=self.len_a)
        self.enco_s = TransEncoder(d_model=d_model, n_head=4, dropout=0.1, dim_ff=2048, num_layers=2,
                                   num_frames=self.len_s)

        self.linear_v = nn.Linear(768, self.d_model)
        self.linear_a = nn.Linear(128, self.d_model)
        self.linear_s = nn.Linear(99, self.d_model)

        self.vlad_v = NeXtVLAD(dim=512, num_clusters=16, lamb=2, groups=8, max_frames=100)
        self.vlad_a = NeXtVLAD(dim=512, num_clusters=16, lamb=2, groups=8, max_frames=100)
        self.vlad_s = NeXtVLAD(dim=512, num_clusters=16, lamb=2, groups=8, max_frames=100)

        self.se = SE(feat_dim=2048 * 3)
        self.classifier = nn.Linear(512, 8)

    def forward(self, feat_v, feat_a, feat_s):
        # visual B x T x d_v -> B x T x d
        #         print('feat_v', type(feat_v))
        feat_v = self.linear_v(feat_v)  # [B, T, 512]
        v_enco = self.enco_v(feat_v)  # [B, T, 512]
        v_agg = self.vlad_v(v_enco)  # [B, 2048]

        # audio B x T x d_a -> B x T x d
        #         print('feat_a', type(feat_a))
        feat_a = self.linear_a(feat_a)  # [B, T, 512]
        a_enco = self.enco_a(feat_a)  # [B, T, 512]
        a_agg = self.vlad_a(a_enco)  # [B, 2048]

        # visual B x T x d_v -> B x T x d
        feat_s = feat_s.reshape([feat_s.shape[0], feat_s.shape[1], -1])
        #         print('feat_s', type(feat_s))
        feat_s = feat_s.to(torch.float32)
        feat_s = self.linear_s(feat_s)  # [B, T, 512]
        s_enco = self.enco_s(feat_s)  # [B, T, 512]
        s_agg = self.vlad_s(s_enco)  # [B, 2048]

        fusion_vector = torch.cat([v_agg, a_agg, s_agg], dim=1)

        fusion = self.se(fusion_vector)

        out = self.classifier(fusion)

        return out

# if __name__ == '__main__':
#
#     ner = NERModel()
#
#     feat_v = torch.randn([32, 100, 768])
#     feat_a = torch.randn([32, 100, 128])
#     feat_s = torch.randn([32, 100, 99])
#     out = ner(feat_v, feat_a, feat_s)
#     print(out.shape)