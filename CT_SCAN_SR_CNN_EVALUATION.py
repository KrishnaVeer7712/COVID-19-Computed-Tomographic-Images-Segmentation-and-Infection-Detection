# -*- coding: utf-8 -*-
"""
Created on Fri Aug 26 23:54:50 2022

@author: JEET
"""

import numpy as np
from tensorflow.keras.layers import  Input, Conv2D, Lambda
from tensorflow.keras.models import Model,load_model
import tensorflow as tf
import tensorflow_model_optimization as tfmot
import nibabel as nb
import scipy.ndimage as ndimage
from skimage.metrics import structural_similarity
from SSIM_MODIFIED import structural_similarity_LCS as SSIM_LCS
import cv2



SCALE_FACTOR = 4

#file_path_train = "J:\\MedicalDataSets\\CT_ORGAN_SEG\\"
file_path_train = 'D:\\E\\Project\\Bakul Gohel Sir\\OrganSeg\\CT_ORGAN_SEG_DATA\\'


def center_crop_local(img, scale_factor=4):
    
    dx,dy,dz= img.shape
    cropx = dx-(dx%scale_factor)
    cropy = dy-(dy%scale_factor)
    startx = dx//2 - cropx//2
    starty = dy//2 - cropy//2    
    return img[startx:startx+cropx, starty:starty+cropy, :]


def read_and_process_nii_1mm(f_no):
    
    file_name_CT_vol = file_path_train + 'volume-' + str(f_no) + '.nii'
    #print(file_name_CT_vol)
    CT_vol_obj = nb.load(file_name_CT_vol)
    x_res = abs(CT_vol_obj.header['srow_x'][0])
    y_res = abs(CT_vol_obj.header['srow_y'][1])
    CT_vol = CT_vol_obj.get_fdata()
    #slice_idx = np.random.randint(0,CT_vol.shape[2]-1,size=BATCH_SIZE)
    #CT_vol = CT_vol[:,:,slice_idx]
    CT_vol = ndimage.zoom(CT_vol,zoom=(x_res, y_res, 1 ), order=3)
    CT_vol = center_crop_local(CT_vol,scale_factor=SCALE_FACTOR)
    Y = np.transpose(CT_vol,(2,0,1))
    Y = np.expand_dims(Y,axis=3)
    Y = (np.clip(Y,a_min=-1024, a_max = 2048)+1024)/2048.0
    X = Y[:,::SCALE_FACTOR,::SCALE_FACTOR,:]
    
    
    file_name_CT_lab = file_path_train + 'labels-' + str(f_no) + '.nii'
    CT_lab_obj = nb.load(file_name_CT_lab)
    CT_lab = CT_lab_obj.get_fdata()
    #CT_lab = CT_lab[:,:,slice_idx]
    CT_lab = ndimage.zoom(CT_lab,zoom=(x_res, y_res, 1 ), order=0)
    CT_lab = center_crop_local(CT_lab,scale_factor=SCALE_FACTOR)
    
    return X,Y,CT_lab


def PSNR_local(D_SR, D_HR, img_bit):
    
    mse = ((D_SR- D_HR) ** 2)
    mse[mse==0] = 100      # MSE is zero means no noise is present in the signal .
                           # Therefore PSNR have no importance.
    max_pixel = 2**img_bit-1
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    return psnr


def SSIM_CT_ORG_local(Y, Y_Pred, CT_mask):
    local_SSIM_all_slices = np.zeros((Y.shape))
    LL_SSIM_all_slices = np.zeros((Y.shape))
    CC_SSIM_all_slices = np.zeros((Y.shape))
    SS_SSIM_all_slices = np.zeros((Y.shape))
    full_slice_ssim_vect =np.zeros((Y.shape[2]))
    local_PSNR_all_slices = np.zeros((Y.shape))
    
    SSIM_MTX_7X4 = np.zeros((7,4))
    PSNR_MTX_7X1 = np.zeros((7,1))

    for i in range(0,Y.shape[2]):
        D_HR = Y[:,:,i]
        D_SR = Y_Pred[:,:,i]
        #ssim, ssim_map = structural_similarity(D_SR, D_HR, win_size = 11, full=True, data_range=(np.max(D_HR)-np.min(D_HR)))
        ssim, ssim_map, LL, CC, SS = SSIM_LCS(D_SR, D_HR, win_size = 11, full=True, data_range=(np.max(D_HR)-np.min(D_HR)))
        full_slice_ssim_vect[i]=ssim
        local_SSIM_all_slices[:,:,i] = ssim_map
        LL_SSIM_all_slices[:,:,i] = LL
        CC_SSIM_all_slices[:,:,i] = CC
        SS_SSIM_all_slices[:,:,i] = SS
        
        local_PSNR_all_slices[:,:,i] = PSNR_local(D_SR*2048, D_HR*2048, img_bit=12)
        
        
    print('Full:',np.mean(full_slice_ssim_vect))    
    
    SSIM_MTX_7X4[0,0] = np.mean(full_slice_ssim_vect)
    SSIM_MTX_7X4[0,1] = np.mean(LL_SSIM_all_slices)
    SSIM_MTX_7X4[0,2] = np.mean(CC_SSIM_all_slices)
    SSIM_MTX_7X4[0,3] = np.mean(SS_SSIM_all_slices)
    
    PSNR_MTX_7X1[0,0] = np.mean(local_PSNR_all_slices)
    

    
    mask = (CT_mask > 0.5)
    SSIM_MTX_7X4[1,0] = np.sum(local_SSIM_all_slices*mask)/np.sum(mask)
    SSIM_MTX_7X4[1,1] = np.sum(LL_SSIM_all_slices*mask)/np.sum(mask)
    SSIM_MTX_7X4[1,2] = np.sum(CC_SSIM_all_slices*mask)/np.sum(mask)
    SSIM_MTX_7X4[1,3] = np.sum(SS_SSIM_all_slices*mask)/np.sum(mask)
    
    PSNR_MTX_7X1[1,0] = np.sum(local_PSNR_all_slices*mask)/np.sum(mask)
    
    
    for k in range(1,6):      
        mask = (np.abs(CT_mask-k)<0.25)
                
        SSIM_MTX_7X4[1+k,0] = np.sum(local_SSIM_all_slices*mask)/np.sum(mask)
        SSIM_MTX_7X4[1+k,1] = np.sum(LL_SSIM_all_slices*mask)/np.sum(mask)
        SSIM_MTX_7X4[1+k,2] = np.sum(CC_SSIM_all_slices*mask)/np.sum(mask)
        SSIM_MTX_7X4[1+k,3] = np.sum(SS_SSIM_all_slices*mask)/np.sum(mask)
        
        PSNR_MTX_7X1[1+k,0] = np.sum(local_PSNR_all_slices*mask)/np.sum(mask)
        
        #print(np.sum(mask), mask.shape, local_SSIM_all_slices.shape)
        #local_ssim = np.sum(local_SSIM_all_slices*mask)/np.sum(mask)
        #print(k, ':' , local_ssim)
    #print(SSIM_MTX_7X4)
    return SSIM_MTX_7X4, PSNR_MTX_7X1

    
def bicubic_SR(X, sf=4):
    # X = (batch, x, y)
    X= np.squeeze(X)
    Y_pred = np.zeros((X.shape[0],X.shape[1]*sf,X.shape[2]*sf))
    for i in range(0,X.shape[0]):
        LR = X[i,:,:]
        Y_pred[i,:,:] = cv2.resize(LR, dsize=(LR.shape[0]*sf,LR.shape[1]*sf), interpolation=cv2.INTER_CUBIC)  
    return Y_pred

def Lanczos_SR(X, sf=4):
    # X = (batch, x, y)
    X= np.squeeze(X)
    Y_pred = np.zeros((X.shape[0],X.shape[1]*sf,X.shape[2]*sf))
    for i in range(0,X.shape[0]):
        LR = X[i,:,:]
        Y_pred[i,:,:] = cv2.resize(LR, dsize=(LR.shape[0]*sf,LR.shape[1]*sf), interpolation=cv2.INTER_LANCZOS4)  
    return Y_pred




###################################### MAIN ####################################################

import pickle
from tensorflow import keras
#model = keras.models.load_model('sr_model_25_mse_v3_SkipConnV2')
model_mse = keras.models.load_model('D:\E\Project\Bakul Gohel Sir\OrganSeg\sr_model_100_mse_v3_SkipConnV2', compile=False)
model_ssim = keras.models.load_model('D:\E\Project\Bakul Gohel Sir\OrganSeg\sr_model_100_ssim_v3_SkipConnV2', compile=False)

#model_mse = keras.models.load_model('D:\E\Project\Bakul Gohel Sir\OrganSeg\sr_model_100_ssim_SkipConnV2_medseg', compile=False)
#model_ssim = keras.models.load_model('D:\E\Project\Bakul Gohel Sir\OrganSeg\sr_model_100_ssim_SkipConnV2_medseg', compile=False)

print('model loaded')

for f_no in range(100,140):
    try:
        print(f_no,'------------------------------------ loading')   
        X,Y,CT_MASK = read_and_process_nii_1mm(f_no)
        print(f_no,'------------------------------------ loaded')
        
        idx = np.random.choice(Y.shape[0],size=150)
        X = X[idx,:,:,:]
        Y = Y[idx,:,:,:]
        
        Y = np.squeeze(Y)
        Y = np.transpose(Y,(1,2,0))   
        CT_MASK  = CT_MASK [:,:,idx]
        
        Y_Pred_bc = bicubic_SR(X, sf=4)
        Y_Pred_bc = np.transpose(np.squeeze(Y_Pred_bc),(1,2,0))
        SSIM_MTX_bc, PSNR_MTX_bc = SSIM_CT_ORG_local(Y, Y_Pred_bc, CT_MASK)
        print('************ Bicubic ***************')
        print(SSIM_MTX_bc)
        print('')
        print(PSNR_MTX_bc)
        
        
        Y_Pred_lc = Lanczos_SR(X, sf=4)
        Y_Pred_lc = np.transpose(np.squeeze(Y_Pred_lc),(1,2,0))
        SSIM_MTX_lc, PSNR_MTX_lc = SSIM_CT_ORG_local(Y, Y_Pred_lc, CT_MASK)
        print('************ LANCZOS ***************')
        print(SSIM_MTX_lc)
        print('')
        print(PSNR_MTX_lc)
        
       
        Y_Pred_mse = model_mse.predict(X)
        Y_Pred_mse = np.transpose(np.squeeze(Y_Pred_mse),(1,2,0))
        SSIM_MTX_mse, PSNR_MTX_mse = SSIM_CT_ORG_local(Y, Y_Pred_mse, CT_MASK)
        print('************ CNN mse loss ***************')
        print(SSIM_MTX_mse)
        print('')
        print(PSNR_MTX_mse)
        
        Y_Pred_ssim = model_ssim.predict(X)
        Y_Pred_ssim = np.transpose(np.squeeze(Y_Pred_ssim),(1,2,0))
        SSIM_MTX_ssim, PSNR_MTX_ssim = SSIM_CT_ORG_local(Y, Y_Pred_ssim, CT_MASK)
        print('************ CNN ssim loss ***************')
        print(SSIM_MTX_ssim)
        print('')
        print(PSNR_MTX_ssim)
        
        print('')
        print('')
        outfile_name = 'D:\E\Project\Bakul Gohel Sir\OrganSeg\CT_ORG_SR_Results\T_ORG_SR_' + str(f_no) + '.pkl'
        out_dict={'SSIM_MTX_bc': SSIM_MTX_bc, 'PSNR_MTX_bc': PSNR_MTX_bc,
                  'SSIM_MTX_lc': SSIM_MTX_lc, 'PSNR_MTX_lc': PSNR_MTX_lc,
                  'SSIM_MTX_mse': SSIM_MTX_mse, 'PSNR_MTX_mse': PSNR_MTX_mse,
                  'SSIM_MTX_ssim': SSIM_MTX_ssim, 'PSNR_MTX_ssim': PSNR_MTX_ssim}
        with open(outfile_name, 'wb') as handle:
            pickle.dump(out_dict, handle)
    except IOError: 
        print(f_no,'...Error Occurred....................................')
        
    
        
