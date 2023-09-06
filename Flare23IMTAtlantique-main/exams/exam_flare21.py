#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import nibabel
import logging
from utils.utils import normalization_imgs, mask_zero
import numpy as np

class exam_flare21:

    def __init__(self, root, id_, scheme:str, anatomy, upload=True):
        
        self.root = root
        self.scheme = scheme
        if self.scheme == 'train':
            self.folder_src = self.root + 'TrainingImg/'
            self.folder_mask = self.root + 'TrainingMask/' 
        elif self.scheme == 'test':
            self.folder_src = self.root
        self.id = id_
        self.anatomy = anatomy
        if upload:
            self.exam_upload()
            self.print_info()
    
    def separate_organs(self):
        
        self.mask_liver = mask_zero(self.mask)
        self.mask_liver.get_fdata(dtype=np.float32)[np.where(self.mask.get_fdata()==1.)] = 1

        self.mask_right_kidneys = mask_zero(self.mask)
        self.mask_right_kidneys.get_fdata(dtype=np.float32)[np.where(self.mask.get_fdata()==2.)] = 1
    
        self.mask_spleen = mask_zero(self.mask)
        self.mask_spleen.get_fdata(dtype=np.float32)[np.where(self.mask.get_fdata()==3.)] = 1
    
        self.mask_pancreas = mask_zero(self.mask)
        self.mask_pancreas.get_fdata(dtype=np.float32)[np.where(self.mask.get_fdata()==4.)] = 1

        self.mask_aorta = mask_zero(self.mask)
        self.mask_aorta.get_fdata(dtype=np.float32)[np.where(self.mask.get_fdata()==5.)] = 1

        self.mask_inferior_vena_cava = mask_zero(self.mask)
        self.mask_inferior_vena_cava.get_fdata(dtype=np.float32)[np.where(self.mask.get_fdata()==6.)] = 1

        self.mask_right_adrenal_gland = mask_zero(self.mask)
        self.mask_right_adrenal_gland.get_fdata(dtype=np.float32)[np.where(self.mask.get_fdata()==7.)] = 1

        self.mask_left_adrenal_gland = mask_zero(self.mask)
        self.mask_left_adrenal_gland.get_fdata(dtype=np.float32)[np.where(self.mask.get_fdata()==8.)] = 1

        self.mask_gallbladder = mask_zero(self.mask)
        self.mask_gallbladder.get_fdata(dtype=np.float32)[np.where(self.mask.get_fdata()==9.)] = 1

        self.mask_esophagus = mask_zero(self.mask)
        self.mask_esophagus.get_fdata(dtype=np.float32)[np.where(self.mask.get_fdata()==10.)] = 1

        self.mask_stomach = mask_zero(self.mask)
        self.mask_stomach.get_fdata(dtype=np.float32)[np.where(self.mask.get_fdata()==11.)] = 1

        self.mask_duodenum = mask_zero(self.mask)
        self.mask_duodenum.get_fdata(dtype=np.float32)[np.where(self.mask.get_fdata()==12.)] = 1

        self.mask_left_kidney = mask_zero(self.mask)
        self.mask_left_kidney.get_fdata(dtype=np.float32)[np.where(self.mask.get_fdata()==13.)] = 1

        self.mask_tumor = mask_zero(self.mask)
        self.mask_tumor.get_fdata(dtype=np.float32)[np.where(self.mask.get_fdata()==14.)] = 1

    def exam_upload(self): 
        
        if self.scheme == 'train':
            self.CT = nibabel.as_closest_canonical(nibabel.load(self.folder_src+'FLARE23_'+self.id+'_0000.nii.gz'))
            self.mask = nibabel.as_closest_canonical(nibabel.load(self.folder_mask+'FLARE23_'+self.id+'.nii.gz'))
            self.separate_organs()
        elif self.scheme == 'test':
            self.CT = nibabel.as_closest_canonical(nibabel.load(self.folder_src+self.id+'_0000.nii.gz'))
            
        
    def normalize(self):
        self.CT.get_fdata()[:,:,:] = normalization_imgs(self.CT.get_fdata())[:,:,:]
            
    def print_info(self):
        
        logging.basicConfig(level=logging.INFO, format='\n %(levelname)s: %(message)s')
        logging.info(f'''exam {self.id} uploaded:                   
        shape:         {self.CT.shape}
        anatomy:       {self.anatomy}
        ''')
