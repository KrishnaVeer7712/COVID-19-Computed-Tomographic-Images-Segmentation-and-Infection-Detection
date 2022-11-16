# -*- coding: utf-8 -*-



import numpy as np
from tensorflow.keras.layers import  Input, Conv2D, Lambda, concatenate , BatchNormalization
from tensorflow.keras.models import Model,load_model
import tensorflow as tf
import tensorflow_model_optimization as tfmot
import nibabel as nb
import scipy.ndimage as ndimage


'''
class espcn:
    def __init__(self, scale_factor=4, image_channels=1,loader=False, qat=True):
        self.__name__ = 'espcn'
        self.scale_factor = scale_factor
        self.channels = image_channels
        self.loader = loader

    # upsampling the resolution of image
    def sub_pixel(self, x):
        return tf.compat.v1.depth_to_space(x, self.scale_factor,name="prediction")
        
    # building the espcn network
    def __call__(self):
        if self.loader is True:
            input_image = Input(shape=(240, 432, self.channels), name='x')
        else:    
            input_image = Input(shape=(None, None, self.channels), name='x')
        x = Conv2D(32, 5, kernel_initializer='glorot_uniform', padding='same', activation=tf.nn.relu,name="conv1")(input_image)
        x = Conv2D(32, 3, kernel_initializer='glorot_uniform', padding='same',activation=tf.nn.relu,name="conv2")(x)
        x = Conv2D(self.scale_factor**2*self.channels, 3, kernel_initializer='glorot_uniform', padding='same',activation=tf.nn.relu,name="conv3")(x)
        if self.scale_factor > 1:
            y = self.sub_pixel(x)
        model = Model(inputs=input_image, outputs=y)
        return model
'''   


BATCH_SIZE = 16
SCALE_FACTOR = 4
file_path_train = 'D:\\E\\Project\\Bakul Gohel Sir\\OrganSeg\\CT_ORGAN_SEG_DATA\\'
#file_path_train = 'E:\\DAIICT_RESEARCH\\BioMedImaging\\DATASETS\\CT_ORGAN_SAMPLE_DATA\\'
#file_path_train = "J:\\MedicalDataSets\\CT_ORGAN_SEG\\"

def center_crop_local(img, scale_factor=4):
    
    dx,dy,dz= img.shape
    cropx = dx-(dx%scale_factor)
    cropy = dy-(dy%scale_factor)
    startx = dx//2 - cropx//2
    starty = dy//2 - cropy//2    
    return img[startx:startx+cropx, starty:starty+cropy, :]


def tissue_specific_slice_slelection(mask, num_slices2select=16):
    area_vect = np.sum(np.sum(mask,axis=0),axis=0)  
    idx = np.squeeze(np.nonzero(area_vect>1000))
    #print('num_index:', idx.shape)
    if idx.size!=0:
        idx = np.random.choice(idx, size=num_slices2select)
    else:
        idx = np.random.choice(np.asarray(range(0,mask.shape[2])), size=num_slices2select)
        
    #print(area_vect[idx])
    return idx
    

def read_and_process_nii_1mm_tissue_specific(f_no, tissue_class = -1):
    
    file_name_CT_vol = file_path_train + 'volume-' + str(f_no) + '.nii'
    CT_vol_obj = nb.load(file_name_CT_vol)
    x_res = abs(CT_vol_obj.header['srow_x'][0])
    y_res = abs(CT_vol_obj.header['srow_y'][1])
    CT_vol = CT_vol_obj.get_fdata()
    
    
    file_name_CT_lab = file_path_train + 'labels-' + str(f_no) + '.nii'
    CT_lab_obj = nb.load(file_name_CT_lab)
    CT_lab = CT_lab_obj.get_fdata()
    
    
    
    if tissue_class is -1:
       mask = (CT_lab>0)
    else:
       mask = (np.abs(CT_lab-tissue_class) < 0.25)
    
       
    slice_idx = tissue_specific_slice_slelection(mask, num_slices2select=BATCH_SIZE)
   
     

    CT_vol = CT_vol[:,:,slice_idx]
    CT_vol = ndimage.zoom(CT_vol,zoom=(x_res, y_res, 1 ), order=3)
    CT_vol = center_crop_local(CT_vol,scale_factor=SCALE_FACTOR)
    CT_vol = (np.clip(CT_vol,a_min=-1024, a_max = 2048)+1024)/2048.0
    
    mask = mask[:,:,slice_idx]
    mask = ndimage.zoom(mask,zoom=(x_res, y_res, 1 ), order=3)
    mask = center_crop_local(mask,scale_factor=SCALE_FACTOR)
    
    CT_vol[np.logical_not(mask)] = -1
    
    #print(CT_vol.shape)
    
    return CT_vol
    
    
    
    


def read_and_process_nii_1mm(f_no, process_label=False):
    file_name_CT_vol = file_path_train + 'volume-' + str(f_no) + '.nii'
    #print(file_name_CT_vol)
    CT_vol_obj = nb.load(file_name_CT_vol)
    x_res = abs(CT_vol_obj.header['srow_x'][0])
    y_res = abs(CT_vol_obj.header['srow_y'][1])
    CT_vol = CT_vol_obj.get_fdata()
    slice_idx = np.random.randint(0,CT_vol.shape[2]-1,size=BATCH_SIZE)
    CT_vol = CT_vol[:,:,slice_idx]
    CT_vol = ndimage.zoom(CT_vol,zoom=(x_res, y_res, 1 ), order=3)
    CT_vol = center_crop_local(CT_vol,scale_factor=SCALE_FACTOR)
    CT_vol = (np.clip(CT_vol,a_min=-1024, a_max = 2048)+1024)/2048.0
    
    
    
    if process_label is True:
        file_name_CT_lab = file_path_train + 'labels-' + str(f_no) + '.nii'
        CT_lab_obj = nb.load(file_name_CT_lab)
        CT_lab = CT_lab_obj.get_fdata()
        CT_lab = CT_lab[:,:,slice_idx]
        CT_lab = ndimage.zoom(CT_lab,zoom=(x_res, y_res, 1 ), order=0)
        CT_lab = center_crop_local(CT_lab,scale_factor=SCALE_FACTOR)
        
        
        
    if process_label is True:
        return CT_vol, CT_lab
    else:
        return CT_vol


def my_sr_CT_org_data_generator():
    while True:
    #for rand_f_no in range(0,10):
        rand_f_no = np.random.randint(21,60)
        CT_vol = read_and_process_nii_1mm(rand_f_no, process_label=False)
        #CT_vol = read_and_process_nii_1mm_tissue_specific(rand_f_no, tissue_class = 3)
        
        Y = np.transpose(CT_vol,(2,0,1))
        Y = np.expand_dims(Y,axis=3)        
        X = Y[:,::SCALE_FACTOR,::SCALE_FACTOR,:]
        #print(X.shape,Y.shape,np.sum(Y))
        #print('X shape:',X.shape)
        #print('Y shape:',Y.shape)
        #print(rand_f_no, X.shape, np.sum(X))
        #print(rand_f_no, Y.shape, np.sum(Y))
        #print('------------------')
        
        yield (X,Y)
#my_sr_CT_org_data_generator()    

def SSIMLoss(y_true, y_pred):
    loss_value =  1 - tf.reduce_mean(tf.image.ssim(y_true, y_pred, 2.0))
    if loss_value is not None:
        return loss_value
    else:
        return 0.0
    
def Custom_MSE(y_true, y_pred):
    W = tf.cast(tf.greater_equal(y_true,0),dtype=float)
    y_true = tf.multiply(y_true, W)
    y_pred = tf.multiply(y_pred, W)   
    
    sd = tf.square(y_true - y_pred)
    sd = tf.reduce_sum(sd, axis=3)
    sd = tf.reduce_sum(sd, axis=2) 
    sd = tf.reduce_sum(sd, axis=1) 
    
    d = tf.reduce_sum(W, axis=3)
    d = tf.reduce_sum(d, axis=2)
    d = tf.reduce_sum(d, axis=1)
      
    #mse = tf.reduce_sum(tf.square(y_true - y_pred))/tf.reduce_sum(W)
    ##mse = tf.reduce_sum(tf.square(y_true - y_pred))
    #mse=0.01
    #return tf.keras.losses.MSE(y_true = y_true, y_pred = y_pred)
    return tf.divide(sd,d)
       
def espcn_model(scale_factor=4, channels=1,loader=False, qat=True):
    input_image = Input(shape=(None, None, channels))
    x1 = Conv2D(32, 7, kernel_initializer='glorot_uniform', padding='same', activation=tf.nn.tanh,name="conv1")(input_image)
    x2 = Conv2D(32, 5, kernel_initializer='glorot_uniform', padding='same',activation=tf.nn.relu,name="conv2")(x1)
    x3 = Conv2D(32, 5, kernel_initializer='glorot_uniform', padding='same',activation=tf.nn.relu,name="conv3")(x2)
    x3 = BatchNormalization()(x3)
    x4 = Conv2D(32, 5, kernel_initializer='glorot_uniform', padding='same',activation=tf.nn.relu,name="conv4")(x3)
    x4 = concatenate([x2,x4], axis = 3)
    x5 = Conv2D(32, 3, kernel_initializer='glorot_uniform', padding='same',activation=tf.nn.tanh,name="conv5")(x4)
    x5 = concatenate([x1,x5], axis = 3)
    x = Conv2D(scale_factor**2*channels, 3, kernel_initializer='glorot_uniform', padding='same',activation='linear',name="convLAST")(x5)
    if scale_factor > 1:
        y = tf.compat.v1.depth_to_space(x, scale_factor,name="prediction")
    model = Model(inputs=input_image, outputs=y)
    return model


 
model = espcn_model()
model.summary()
"""
# define U-Net model architecture

def build_unet(img_shape):
    # input layer shape is equal to patch image size
    inputs = Input(shape=img_shape)

    # rescale images from (0, 255) to (0, 1)
    rescale = Rescaling(scale=1. / 255, input_shape=(img_height, img_width, img_channels))(inputs)
    previous_block_activation = rescale  # Set aside residual

    contraction = {}
    # # Contraction path: Blocks 1 through 5 are identical apart from the feature depth
    for f in [16, 32, 64, 128]:
        x = Conv2D(f, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(previous_block_activation)
        x = Dropout(0.1)(x)
        x = Conv2D(f, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(x)
        contraction[f'conv{f}'] = x
        x = MaxPooling2D((2, 2))(x)
        previous_block_activation = x

    c5 = Conv2D(160, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(previous_block_activation)
    c5 = Dropout(0.2)(c5)
    c5 = Conv2D(160, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c5)
    previous_block_activation = c5

    # Expansive path: Second half of the network: upsampling inputs
    for f in reversed([16, 32, 64, 128]):
        x = Conv2DTranspose(f, (2, 2), strides=(2, 2), padding='same')(previous_block_activation)
        x = concatenate([x, contraction[f'conv{f}']])
        x = Conv2D(f, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(x)
        x = Dropout(0.2)(x)
        x = Conv2D(f, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(x)
        previous_block_activation = x

    outputs = Conv2D(filters=n_classes, kernel_size=(1, 1), activation="softmax")(previous_block_activation)

    return Model(inputs=inputs, outputs=outputs)


## build model
#model2 = build_unet(img_shape=(img_height, img_width, img_channels))
#model2 = build_unet(img_shape=(None, None, 1))
#model2.summary()
"""
#sgd = tf.optimizers.SGD(lr=0.00001, decay=1e-6, momentum=0.9, nesterov=False,clipnorm=100)
ADAM = tf.optimizers.Adam(learning_rate=0.001, clipnorm=1., clipvalue=0.5)
#model.compile(loss='mean_squared_error', optimizer=ADAM)
model.compile(loss=SSIMLoss, optimizer=ADAM)






if False:

    model_name = 'D:\E\Project\Bakul Gohel Sir\OrganSeg\sr_model_25_mse_v3_SkipConnV2_C3'
    
    
    
    model.fit(my_sr_CT_org_data_generator(), validation_data = my_sr_CT_org_data_generator(),
              steps_per_epoch = 5,
              validation_steps=1,
              epochs = 25 )
    
    
    model.save(model_name)



if True:
    #model_name = 'sr_model_25_mse_v3_SkipConnV2_C3'
    
    file_in = 'D:\\E\\Project\\Bakul Gohel Sir\\covid-segmentation\\images_medseg.npy'
    CT_img = np.load(file_in)/255.0
    CT_img_train = CT_img[0:70,:,:,:]
    CT_img_test = CT_img[70:100,:,:,:]
    
    model.fit(CT_img_train[:,::SCALE_FACTOR,::SCALE_FACTOR,:], CT_img_train,
              validation_data=(CT_img_test[:,::SCALE_FACTOR,::SCALE_FACTOR,:], CT_img_test),
              epochs=5, batch_size=4) #epochs 100

    model_out_name = 'D:\E\Project\Bakul Gohel Sir\OrganSeg\sr_model_100_ssim_SkipConnV2_medseg'
    model.save(model_out_name)

#file_path = 'E:\\DAIICT_RESEARCH\\BioMedImaging\\DATASETS\\CT_ORGAN_SAMPLE_DATA\\'

