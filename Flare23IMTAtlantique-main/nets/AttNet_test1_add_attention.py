#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
from torchvision import models
from torch import nn
from nets import block

class uNet13DS(nn.Module):
    
    def __init__(self, n_classes, pretrained=True):    
        super(uNet13DS, self).__init__()
        self.n_channels = 3
        self.n_classes = n_classes
        
        self.pool = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU(inplace=True)
        self.encoder = models.vgg13_bn(pretrained=pretrained).features
        self.enblock1 = nn.Sequential(self.encoder[0], self.encoder[1], self.relu, self.encoder[3], self.encoder[4], self.relu)
        self.enblock2 = nn.Sequential(self.pool, self.encoder[7], self.encoder[8], self.relu, self.encoder[10], self.encoder[11], self.relu)
        self.enblock3 = nn.Sequential(self.pool, self.encoder[14], self.encoder[15], self.relu, self.encoder[17], self.encoder[18], self.relu)
        self.enblock4 = nn.Sequential(self.pool, self.encoder[21], self.encoder[22], self.relu, self.encoder[24], self.encoder[25], self.relu)
        self.center = nn.Sequential(self.pool, self.encoder[28], self.encoder[29], self.relu, self.encoder[31], self.encoder[32], self.relu) ## 瓶颈层结构（可以尝试修改）
        
        self.decoder_att1 = block.Dec_attention1(512,512)
        self.decoder_att2 = block.Dec_attention2(512,256)
        self.decoder_att3 = block.Dec_attention2(256,128)
        self.decoder_att4 = block.Dec_attention3(128,64)

        self.down4 = block.SimpleConv(512, 16)
        self.down3 = block.SimpleConv(256, 16)
        self.down2 = block.SimpleConv(128, 16)
        self.down1 = block.SimpleConv(64, 16)
        
        self.down_out1 = block.SingleConv(16, self.n_classes)
        self.down_out2 = block.SingleConv(16, self.n_classes)
        self.down_out3 = block.SingleConv(16, self.n_classes)
        self.down_out4 = block.SingleConv(16, self.n_classes)
        
        self.out1 = block.SimpleConv(16*4, 64)##合并4个提炼特征层的层
        self.att = block.attention()
        self.out2 = block.SingleConv(64, self.n_classes)## conv+softmax层（注意力机制可以添加在合并层和softmax之间，以便于让softmax层能更好的对像素级标签进行预测）

    def forward(self, x):
        x1 = self.enblock1(x)
        x2 = self.enblock2(x1)
        x3 = self.enblock3(x2)
        x4 = self.enblock4(x3)
        x5 = self.center(x4)

        dec4 = self.decoder_att1(x5, x4)## 瓶颈层后的“up conv”层
        dec3 = self.decoder_att2(dec4, x3)
        dec2 = self.decoder_att3(dec3, x2)
        dec1 = self.decoder_att4(dec2, x1)

        down1 = block.up_sample2d(self.down1(dec1), x)
        down2 = block.up_sample2d(self.down2(dec2), x)
        down3 = block.up_sample2d(self.down3(dec3), x)
        down4 = block.up_sample2d(self.down4(dec4), x)

        down_out1 = self.down_out1(down1)
        down_out2 = self.down_out2(down2)
        down_out3 = self.down_out3(down3)
        down_out4 = self.down_out4(down4)
        
        logits_out1 = self.out1(torch.cat((down1, down2, down3, down4), dim=1))
        logits_attention = self.att(logits_out1)
        logits = self.out2(logits_attention)
        
        if self.training:
            return logits, down_out1, down_out2, down_out3, down_out4
        else:
            return logits