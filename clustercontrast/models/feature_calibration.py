# -*- coding: UTF-8 -*-
'''
@Project ：TransReID-SSL-main-v3
@File ：feature_calibration.py
@Author ：棋子
@Date ：2023/7/12 19:02
@Software: PyCharm
'''

import torch
import torch.nn as nn
import torch.nn.functional as F


class FCM(nn.Module):
    def __init__(self, temperature=1):
        super(FCM, self).__init__()
        self.temperature = temperature

    def forward(self, feats, k=16, w=1):
        print('======== fcm  input feats ==========')
        # print(feats[0, 0:10])
        scores = torch.matmul(feats, feats.transpose(-2, -1)) / self.temperature     # [256, 256]
        # print(scores.shape)

        # 保留每一行的最大的8个值，其余置为 inf
        top_k_values, top_k_indices = torch.topk(scores, k=k, dim=1)  # 在每一行中找到最大的k个值和对应的索引


        # 创建一个与score相同形状的张量，填充为负无穷
        result = torch.full_like(scores, float("-inf"))
        # 使用索引将top_k_values复制到result中
        result.scatter_(1, top_k_indices, top_k_values)
        # result.scatter_(1, top_k_indices[:, 1:], top_k_values[:, 1:])
        # print(result[:10])

        attn_weights = nn.Softmax(dim=-1)(result)
        feats_att = torch.matmul(attn_weights, feats)
        feats_fcm = feats*w + feats_att
        # feats_fcm = feats_att
        feats_norm = F.normalize(feats_fcm, p=2, dim=1)

        return feats_norm