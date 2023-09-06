#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from nets import AttNet_test1_add_attention

def whichnet(net_id, n_classes):
    
    if net_id == 1:
        pretrained, vgg = False, True
        net = AttNet_test1_add_attention.uNet13DS(n_classes = n_classes, pretrained = pretrained)
    
    return net, vgg