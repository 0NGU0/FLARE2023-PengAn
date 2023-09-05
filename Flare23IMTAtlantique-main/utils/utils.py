
import numpy as np
from skimage.segmentation import mark_boundaries
from skimage.exposure import rescale_intensity
from skimage.measure import label
import nibabel

def boundaries(img, pred, groundtruth):
    img = rescale_intensity(img, in_range=(np.min(img),np.max(img)), out_range=(0,1))

    if type(pred) == np.ndarray and type(groundtruth) == np.ndarray:
        out = mark_boundaries(img, groundtruth, color=(0, 1, 0), background_label=4)

        out = mark_boundaries(out, pred, color=(1, 0, 0), background_label=2)
    else:
        if type(pred) == np.ndarray:
            out = mark_boundaries(img, pred, color=(1, 0, 0), background_label=2)
        if type(groundtruth) == np.ndarray:
            out = mark_boundaries(img, groundtruth, color=(0, 1, 0), background_label=4)            
    return out
   
def normalization_imgs(imgs):
    ''' centering and reducing data structures '''
    imgs = imgs.astype(np.float32, copy=False)
    mean = np.mean(imgs) 
    std = np.std(imgs)
    if np.int32(std) != 0:
        imgs -= mean
        imgs /= std
    return imgs

def get_array_affine_header(test_dataset, modality):
    if modality == 'T2':
        array = np.zeros(test_dataset.exam.T2.shape, dtype=np.uint16)
        affine, header = test_dataset.exam.T2.affine, test_dataset.exam.T2.header
    elif modality == 'CT':
        array = np.zeros(test_dataset.exam.CT.shape, dtype=np.uint16)
        affine, header = test_dataset.exam.CT.affine, test_dataset.exam.CT.header        
    return array, affine, header

def mask_zero(mask):
    return nibabel.Nifti1Image(np.zeros(shape=mask.shape).astype(np.uint8), affine=mask.affine, header=mask.header)

def get_largest_connected_region(segmentation):
    if len(np.unique(segmentation)) == 1:

        return segmentation
    else:
        labels = label(segmentation,connectivity=1)

        unique, counts = np.unique(labels, return_counts=True)
        list_seg=list(zip(unique, counts))[1:] 

        largest=max(list_seg, key=lambda x:x[1])[0]
        labels_max=(labels == largest).astype(int)
        return labels_max
    
def get_2_largest_connected_region(segmentation):
    if len(np.unique(segmentation)) == 1:
        return segmentation
    else:
        labels = label(segmentation,connectivity=1)
        unique, counts = np.unique(labels, return_counts=True)
        list_seg=list(zip(unique, counts))[1:] 
        largest=max(list_seg, key=lambda x:x[1])[0]
        labels_max1=(labels == largest)
        labels[np.where(labels == largest)] = 0
        unique, counts = np.unique(labels, return_counts=True)
        list_seg=list(zip(unique, counts))[1:] 
        largest=max(list_seg, key=lambda x:x[1])[0]
        labels_max2=(labels == largest)
        return labels_max1+labels_max2