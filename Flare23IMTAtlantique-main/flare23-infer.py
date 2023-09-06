
import logging
import numpy as np
import torch
from utils.utils import get_array_affine_header, get_largest_connected_region
from torch.utils.data import DataLoader
import tqdm
import distutils.dir_util
import nibabel
from skimage.transform import resize, rotate
from datasets.dataset_flare21 import tiny_dataset_flare21
from nets.whichnet import whichnet
import os

def listdir_nohidden(path):
    l = []
    for f in np.sort(os.listdir(path)) :

        if f.startswith('.') == False :

            l.append(f)
    return l

def infer_flare21(net,
                  net_id,
                  anatomy,
                  output,
                  device,
                  vgg,
                  size):

    list_ = listdir_nohidden('./inputs')

    test_ids = []
    for elem_ in list_:
        if elem_.split('.')[1] == 'nii':
            test_ids.append(elem_.split('_0000.nii.gz')[0])

    for index, id_ in enumerate(tqdm.tqdm(test_ids)):
        
        test_dataset = tiny_dataset_flare21(id_, size, anatomy, vgg)
        
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
        
        array, affine, header = get_array_affine_header(test_dataset, 'CT')
        
        array_liver = np.copy(array)
        array_right_kidneys = np.copy(array)
        array_spleen = np.copy(array)
        array_pancreas = np.copy(array)  
        array_aorta = np.copy(array)  
        array_inferior_vena_cava = np.copy(array)  
        array_right_adrenal_gland = np.copy(array)  
        array_left_adrenal_gland = np.copy(array)  
        array_gallbladder = np.copy(array)  
        array_esophagus = np.copy(array)  
        array_stomach = np.copy(array)  
        array_duodenum = np.copy(array)  
        array_left_kidney = np.copy(array)  
        array_tumor = np.copy(array)  


        with torch.no_grad():
            
            for idx, data in enumerate(test_loader):
                
                image = data
                image = image.to(device=device, dtype=torch.float32)
                
                net.training = False

                prob = torch.softmax(net(image), dim=1)

                pred = torch.argmax(prob, dim=1).float()

                
                full_mask = pred.squeeze().cpu().numpy().swapaxes(0,1).astype(np.uint8)


                mask_liver = np.zeros(shape=full_mask.shape, dtype=np.uint8)
                mask_right_kidneys = np.zeros(shape=full_mask.shape, dtype=np.uint8)
                mask_spleen = np.zeros(shape=full_mask.shape, dtype=np.uint8)
                mask_pancreas = np.zeros(shape=full_mask.shape, dtype=np.uint8)
                mask_aorta = np.zeros(shape=full_mask.shape, dtype=np.uint8)
                mask_inferior_vena_cava = np.zeros(shape=full_mask.shape, dtype=np.uint8)
                mask_right_adrenal_gland = np.zeros(shape=full_mask.shape, dtype=np.uint8)
                mask_left_adrenal_gland = np.zeros(shape=full_mask.shape, dtype=np.uint8)
                mask_gallbladder = np.zeros(shape=full_mask.shape, dtype=np.uint8)
                mask_esophagus = np.zeros(shape=full_mask.shape, dtype=np.uint8)
                mask_stomach = np.zeros(shape=full_mask.shape, dtype=np.uint8)
                mask_duodenum = np.zeros(shape=full_mask.shape, dtype=np.uint8)
                mask_left_kidney = np.zeros(shape=full_mask.shape, dtype=np.uint8)
                mask_tumor = np.zeros(shape=full_mask.shape, dtype=np.uint8)

                mask_liver[np.where(full_mask==1)] = 1
                mask_right_kidneys[np.where(full_mask==2)] = 1
                mask_spleen[np.where(full_mask==3)] = 1
                mask_pancreas[np.where(full_mask==4)] = 1              
                mask_aorta[np.where(full_mask==5)] = 1
                mask_inferior_vena_cava[np.where(full_mask==6)] = 1
                mask_right_adrenal_gland[np.where(full_mask==7)] = 1
                mask_left_adrenal_gland[np.where(full_mask==8)] = 1
                mask_gallbladder[np.where(full_mask==9)] = 1
                mask_esophagus[np.where(full_mask==10)] = 1
                mask_stomach[np.where(full_mask==11)] = 1
                mask_duodenum[np.where(full_mask==12)] = 1
                mask_left_kidney[np.where(full_mask==13)] = 1
                mask_tumor[np.where(full_mask==14)] = 1

                mask_liver = resize(rotate(mask_liver, 90, preserve_range=True), output_shape=(test_dataset.exam.CT.shape[0],test_dataset.exam.CT.shape[1]), preserve_range=True)
                mask_right_kidneys = resize(rotate(mask_right_kidneys, 90, preserve_range=True), output_shape=(test_dataset.exam.CT.shape[0],test_dataset.exam.CT.shape[1]), preserve_range=True)
                mask_spleen = resize(rotate(mask_spleen, 90, preserve_range=True), output_shape=(test_dataset.exam.CT.shape[0],test_dataset.exam.CT.shape[1]), preserve_range=True)
                mask_pancreas = resize(rotate(mask_pancreas, 90, preserve_range=True), output_shape=(test_dataset.exam.CT.shape[0],test_dataset.exam.CT.shape[1]), preserve_range=True)
                mask_aorta = resize(rotate(mask_aorta, 90, preserve_range=True), output_shape=(test_dataset.exam.CT.shape[0],test_dataset.exam.CT.shape[1]), preserve_range=True)
                mask_inferior_vena_cava = resize(rotate(mask_inferior_vena_cava, 90, preserve_range=True), output_shape=(test_dataset.exam.CT.shape[0],test_dataset.exam.CT.shape[1]), preserve_range=True)
                mask_right_adrenal_gland = resize(rotate(mask_right_adrenal_gland, 90, preserve_range=True), output_shape=(test_dataset.exam.CT.shape[0],test_dataset.exam.CT.shape[1]), preserve_range=True)
                mask_left_adrenal_gland = resize(rotate(mask_left_adrenal_gland, 90, preserve_range=True), output_shape=(test_dataset.exam.CT.shape[0],test_dataset.exam.CT.shape[1]), preserve_range=True)
                mask_gallbladder = resize(rotate(mask_gallbladder, 90, preserve_range=True), output_shape=(test_dataset.exam.CT.shape[0],test_dataset.exam.CT.shape[1]), preserve_range=True)
                mask_esophagus = resize(rotate(mask_esophagus, 90, preserve_range=True), output_shape=(test_dataset.exam.CT.shape[0],test_dataset.exam.CT.shape[1]), preserve_range=True)
                mask_stomach = resize(rotate(mask_stomach, 90, preserve_range=True), output_shape=(test_dataset.exam.CT.shape[0],test_dataset.exam.CT.shape[1]), preserve_range=True)
                mask_duodenum = resize(rotate(mask_duodenum, 90, preserve_range=True), output_shape=(test_dataset.exam.CT.shape[0],test_dataset.exam.CT.shape[1]), preserve_range=True)
                mask_left_kidney = resize(rotate(mask_left_kidney, 90, preserve_range=True), output_shape=(test_dataset.exam.CT.shape[0],test_dataset.exam.CT.shape[1]), preserve_range=True)
                mask_tumor = resize(rotate(mask_tumor, 90, preserve_range=True), output_shape=(test_dataset.exam.CT.shape[0],test_dataset.exam.CT.shape[1]), preserve_range=True)

                mask_liver[np.where(mask_liver>0.95)] = 1
                mask_liver[np.where(mask_liver!=1)] = 0
                
                mask_right_kidneys[np.where(mask_right_kidneys>0.95)] = 1
                mask_right_kidneys[np.where(mask_right_kidneys!=1)] = 0
                
                mask_spleen[np.where(mask_spleen>0.95)] = 1
                mask_spleen[np.where(mask_spleen!=1)] = 0
                
                mask_pancreas[np.where(mask_pancreas>0.95)] = 1
                mask_pancreas[np.where(mask_pancreas!=1)] = 0

                mask_aorta[np.where(mask_aorta>0.95)] = 1
                mask_aorta[np.where(mask_aorta!=1)] = 0

                mask_inferior_vena_cava[np.where(mask_inferior_vena_cava>0.95)] = 1
                mask_inferior_vena_cava[np.where(mask_inferior_vena_cava!=1)] = 0

                mask_right_adrenal_gland[np.where(mask_right_adrenal_gland>0.95)] = 1
                mask_right_adrenal_gland[np.where(mask_right_adrenal_gland!=1)] = 0

                mask_left_adrenal_gland[np.where(mask_left_adrenal_gland>0.95)] = 1
                mask_left_adrenal_gland[np.where(mask_left_adrenal_gland!=1)] = 0

                mask_gallbladder[np.where(mask_gallbladder>0.95)] = 1
                mask_gallbladder[np.where(mask_gallbladder!=1)] = 0

                mask_esophagus[np.where(mask_esophagus>0.95)] = 1
                mask_esophagus[np.where(mask_esophagus!=1)] = 0

                mask_stomach[np.where(mask_stomach>0.95)] = 1
                mask_stomach[np.where(mask_stomach!=1)] = 0

                mask_duodenum[np.where(mask_duodenum>0.95)] = 1
                mask_duodenum[np.where(mask_duodenum!=1)] = 0

                mask_left_kidney[np.where(mask_left_kidney>0.95)] = 1
                mask_left_kidney[np.where(mask_left_kidney!=1)] = 0

                mask_tumor[np.where(mask_tumor>0.95)] = 1
                mask_tumor[np.where(mask_tumor!=1)] = 0

                array_liver[0:test_dataset.exam.CT.shape[0],0:test_dataset.exam.CT.shape[1],idx] = mask_liver[::-1,::]
                array_right_kidneys[0:test_dataset.exam.CT.shape[0],0:test_dataset.exam.CT.shape[1],idx] = mask_right_kidneys[::-1,::]
                array_spleen[0:test_dataset.exam.CT.shape[0],0:test_dataset.exam.CT.shape[1],idx] = mask_spleen[::-1,::]
                array_pancreas[0:test_dataset.exam.CT.shape[0],0:test_dataset.exam.CT.shape[1],idx] = mask_pancreas[::-1,::]
                array_aorta[0:test_dataset.exam.CT.shape[0],0:test_dataset.exam.CT.shape[1],idx] = mask_aorta[::-1,::]
                array_inferior_vena_cava[0:test_dataset.exam.CT.shape[0],0:test_dataset.exam.CT.shape[1],idx] = mask_inferior_vena_cava[::-1,::]
                array_right_adrenal_gland[0:test_dataset.exam.CT.shape[0],0:test_dataset.exam.CT.shape[1],idx] = mask_right_adrenal_gland[::-1,::]
                array_left_adrenal_gland[0:test_dataset.exam.CT.shape[0],0:test_dataset.exam.CT.shape[1],idx] = mask_left_adrenal_gland[::-1,::]
                array_gallbladder[0:test_dataset.exam.CT.shape[0],0:test_dataset.exam.CT.shape[1],idx] = mask_gallbladder[::-1,::]
                array_esophagus[0:test_dataset.exam.CT.shape[0],0:test_dataset.exam.CT.shape[1],idx] = mask_esophagus[::-1,::]
                array_stomach[0:test_dataset.exam.CT.shape[0],0:test_dataset.exam.CT.shape[1],idx] = mask_stomach[::-1,::]
                array_duodenum[0:test_dataset.exam.CT.shape[0],0:test_dataset.exam.CT.shape[1],idx] = mask_duodenum[::-1,::]
                array_left_kidney[0:test_dataset.exam.CT.shape[0],0:test_dataset.exam.CT.shape[1],idx] = mask_left_kidney[::-1,::]
                array_tumor[0:test_dataset.exam.CT.shape[0],0:test_dataset.exam.CT.shape[1],idx] = mask_tumor[::-1,::]
                


        array_liver = get_largest_connected_region(array_liver)
        array_right_kidneys = get_largest_connected_region(array_right_kidneys)
        array_spleen = get_largest_connected_region(array_spleen)
        array_pancreas = get_largest_connected_region(array_pancreas)
        array_aorta = get_largest_connected_region(array_aorta)
        array_inferior_vena_cava = get_largest_connected_region(array_inferior_vena_cava)
        array_right_adrenal_gland = get_largest_connected_region(array_right_adrenal_gland)
        array_left_adrenal_gland = get_largest_connected_region(array_left_adrenal_gland)
        array_gallbladder = get_largest_connected_region(array_gallbladder)
        array_esophagus = get_largest_connected_region(array_esophagus)
        array_stomach = get_largest_connected_region(array_stomach)
        array_duodenum = get_largest_connected_region(array_duodenum)
        array_left_kidney = get_largest_connected_region(array_left_kidney)
        array_tumor = get_largest_connected_region(array_tumor)

        array[np.where(array_liver==1)] = 1
        array[np.where(array_right_kidneys==1)] = 2
        array[np.where(array_spleen==1)] = 3
        array[np.where(array_pancreas==1)] = 4
        array[np.where(array_aorta==1)] = 5
        array[np.where(array_inferior_vena_cava==1)] = 6
        array[np.where(array_right_adrenal_gland==1)] = 7
        array[np.where(array_left_adrenal_gland==1)] = 8
        array[np.where(array_gallbladder==1)] = 9
        array[np.where(array_esophagus==1)] = 10
        array[np.where(array_stomach==1)] = 11
        array[np.where(array_duodenum==1)] = 12
        array[np.where(array_left_kidney==1)] = 13
        array[np.where(array_tumor==1)] = 14


        
        prediction = nibabel.Nifti1Image(array.astype(np.uint8), affine=affine)
        nibabel.save(prediction, './outputs/'+id_+'.nii.gz')


        del prediction, test_dataset, array, array_liver, array_right_kidneys, array_spleen, array_pancreas,array_aorta,array_inferior_vena_cava,array_right_adrenal_gland,array_left_adrenal_gland,array_gallbladder,array_esophagus,array_stomach,array_duodenum,array_left_kidney,array_tumor

if __name__ == "__main__":


    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
        
    model = './Flare23IMTAtlantique-main/weights/epoch.pth'
    
    output = './outputs'
    
    net_id = 1
    
    n_classes = 15 
    
    size = 512   

    net, vgg = whichnet(net_id, n_classes)

    logging.info("loading model {}".format(model))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    logging.info(f'using device {device}')
    
    net.to(device=device)
    net.load_state_dict(torch.load(model, map_location=device))

    logging.info("model loaded !")

    infer_flare21(net = net,
                  net_id = net_id,
                  anatomy = 'all',
                  output = output, 
                  device = device,
                  vgg = vgg,
                  size = size)
