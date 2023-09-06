#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from exams.exam_flare21 import exam_flare21
import distutils.dir_util
from skimage import img_as_ubyte
from skimage.exposure import rescale_intensity
from skimage.transform import resize, rotate
from skimage import io
import tqdm
from utils import utils
import random

def create_flare21_dataset(root, output, ids, scheme, size, anatomy):
    ''' create flare21 dataset with scheme={train,val} and anatomy={all, liver, kidneys, spleen, pancreas} '''
    
    list_id = []
    folder = output+scheme+'/'
    distutils.dir_util.mkpath(folder)
    
    for idx, id_ in enumerate(tqdm.tqdm(ids)):
        exam = exam_flare21(root, id_, 'train', anatomy)
        exam.normalize()
        for xyz in range(exam.CT.shape[2]): 
            img, mask = extract_flare21_slice(exam, xyz, size, anatomy)
            if len(np.unique(img))>1: # to avoid blank images
                list_id.append('CT-%0*d-'%(4,id_)+anatomy+'-%0*d'%(4,xyz+1))
                io.imsave(folder+list_id[-1]+'-src.png', img)

                io.imsave(folder+list_id[-1]+'-mask.png', mask, check_contrast=False)
        del exam

    np.save(output+'imgs-id-'+scheme+'.npy', list_id)
    del list_id
    
def extract_flare21_slice(exam, xyz, size, anatomy, mask_available=True):
    
    img = rotate(resize(np.squeeze(exam.CT.get_fdata()[:,:,xyz])[::-1,:], output_shape=(size,size), preserve_range=True), 90, preserve_range=True)

    min_greyscale, max_greyscale = np.percentile(img,(1,99)) 

    img = rescale_intensity(img, in_range=(min_greyscale,max_greyscale), out_range=(0,1))

    img = img.astype(np.float32)
    
    if mask_available:
        if anatomy == 'all':
            mask_liver = rotate(resize(exam.mask_liver.get_fdata(dtype=np.float32)[:,:,xyz][::-1,:]*255, output_shape=(size,size), preserve_range=True), 90, preserve_range=True)        

            mask_right_kidneys = rotate(resize(exam.mask_right_kidneys.get_fdata(dtype=np.float32)[:,:,xyz][::-1,:]*255, output_shape=(size,size), preserve_range=True), 90, preserve_range=True)

            mask_spleen = rotate(resize(exam.mask_spleen.get_fdata(dtype=np.float32)[:,:,xyz][::-1,:]*255, output_shape=(size,size), preserve_range=True), 90, preserve_range=True)

            mask_pancreas = rotate(resize(exam.mask_pancreas.get_fdata(dtype=np.float32)[:,:,xyz][::-1,:]*255, output_shape=(size,size), preserve_range=True), 90, preserve_range=True)

            mask_aorta = rotate(resize(exam.mask_aorta.get_fdata(dtype=np.float32)[:,:,xyz][::-1,:]*255, output_shape=(size,size), preserve_range=True), 90, preserve_range=True)

            mask_inferior_vena_cava = rotate(resize(exam.mask_inferior_vena_cava.get_fdata(dtype=np.float32)[:,:,xyz][::-1,:]*255, output_shape=(size,size), preserve_range=True), 90, preserve_range=True)

            mask_right_adrenal_gland = rotate(resize(exam.mask_right_adrenal_gland.get_fdata(dtype=np.float32)[:,:,xyz][::-1,:]*255, output_shape=(size,size), preserve_range=True), 90, preserve_range=True)

            mask_left_adrenal_gland = rotate(resize(exam.mask_left_adrenal_gland.get_fdata(dtype=np.float32)[:,:,xyz][::-1,:]*255, output_shape=(size,size), preserve_range=True), 90, preserve_range=True)

            mask_gallbladder = rotate(resize(exam.mask_gallbladder.get_fdata(dtype=np.float32)[:,:,xyz][::-1,:]*255, output_shape=(size,size), preserve_range=True), 90, preserve_range=True)

            mask_esophagus = rotate(resize(exam.mask_esophagus.get_fdata(dtype=np.float32)[:,:,xyz][::-1,:]*255, output_shape=(size,size), preserve_range=True), 90, preserve_range=True)

            mask_stomach = rotate(resize(exam.mask_stomach.get_fdata(dtype=np.float32)[:,:,xyz][::-1,:]*255, output_shape=(size,size), preserve_range=True), 90, preserve_range=True)

            mask_duodenum = rotate(resize(exam.mask_duodenum.get_fdata(dtype=np.float32)[:,:,xyz][::-1,:]*255, output_shape=(size,size), preserve_range=True), 90, preserve_range=True)

            mask_left_kidney = rotate(resize(exam.mask_left_kidney.get_fdata(dtype=np.float32)[:,:,xyz][::-1,:]*255, output_shape=(size,size), preserve_range=True), 90, preserve_range=True)

            mask_tumor = rotate(resize(exam.mask_tumor.get_fdata(dtype=np.float32)[:,:,xyz][::-1,:]*255, output_shape=(size,size), preserve_range=True), 90, preserve_range=True)

        if anatomy == 'all':
            mask = np.zeros(shape=mask_liver.shape, dtype=np.uint8)
            mask[np.where(mask_liver>0)] = 1
            mask[np.where(mask_right_kidneys>0)] = 2
            mask[np.where(mask_spleen>0)] = 3
            mask[np.where(mask_pancreas>0)] = 4
            mask[np.where(mask_aorta>0)] = 5
            mask[np.where(mask_inferior_vena_cava>0)] = 6
            mask[np.where(mask_right_adrenal_gland>0)] = 7
            mask[np.where(mask_left_adrenal_gland>0)] = 8
            mask[np.where(mask_gallbladder>0)] = 9
            mask[np.where(mask_esophagus>0)] = 10
            mask[np.where(mask_stomach>0)] = 11
            mask[np.where(mask_duodenum>0)] = 12
            mask[np.where(mask_left_kidney>0)] = 13
            mask[np.where(mask_tumor>0)] = 14
            return img_as_ubyte(img), mask.astype(np.uint8)
        
    else:
        return img_as_ubyte(img)

def flare21_split():
    
    ids = list(range(0,2200))
    random.seed(4)
    random.shuffle(ids)
    train_ids = ids[:1980]
    val_ids = ids[1980:]
    return list(train_ids), list(val_ids)