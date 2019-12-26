
# coding: utf-8

# In[1]:


#Import


from keras import backend as K
from keras.utils.generic_utils import get_custom_objects
import os
#os.environ["CUDA_VISIBLE_DEVICES"]="-1" 
import numpy as np
import keras
import scipy as sc
from keras.layers import Dense,Conv2D,Conv2DTranspose,Flatten,Input,concatenate,Reshape,Dropout,Activation,BatchNormalization,MaxPooling2D,UpSampling2D
from keras.optimizers import SGD
from keras.callbacks import CSVLogger,TerminateOnNaN,EarlyStopping,ModelCheckpoint
import scipy.io
from scipy.io import loadmat
from keras.models import Model
from keras import regularizers
import csv
from keras.models import load_model
from keras.models import model_from_json

class SWISH(Activation):

        def __init__(self,activation,**kwargs):
                super(SWISH,self).__init__(activation,**kwargs)
                self.__name__='swish_fn'

def swish(x):
        return (x*K.sigmoid(x))

get_custom_objects().update({'swish_fn': SWISH(swish)})


class CNNCFD():
    def __init__(self):
        self.x_train_1=loadmat('SDF_values.mat')['SDF_values']
        self.x_train_2=loadmat('input_re_alpha.mat')['input_re_alpha']    #Load training data
        self.y_train_1=loadmat('Output_values.mat')['Output_values']

        self.y_train_2=loadmat('Output_u.mat')['Output_u']
        self.y_train_3=loadmat('Output_v.mat')['Output_v']
        self.y_train_4=loadmat('Output_p.mat')['Output_p']
        
        self.y_train_2=np.expand_dims(self.y_train_2,axis=3)
        self.y_train_3=np.expand_dims(self.y_train_3,axis=3)
        self.y_train_4=np.expand_dims(self.y_train_4,axis=3)

        self.x_train_1 = np.expand_dims(self.x_train_1,axis=3)
        self.x_train_2 = np.expand_dims(self.x_train_2,axis=2)
        pass
    def load_data(self):
        return
    def build_model(self,num_conv,num_dense,inp_optim,inp_loss):
        return model
    def train_model(self,batch_sz,eps,val_splt,model):
        return model
    def plot_error(self,exp_no):
        return
    def model_prediction(self,x1,x2):
        return ans
    def save_model(self,model,modelname):
        return
    def load_model(self,modelname):
        return model
    


CNNCFD_instance= CNNCFD()

keras.optimizers.Adam(lr=0.001,decay=0.5)

def load_data():
        x_train_1=loadmat('SDF_values.mat')['SDF_values']
        x_train_2=loadmat('input_re_alpha.mat')['input_re_alpha']    #Load training data
        y_train_1=loadmat('Output_values.mat')['Output_values']

        y_train_2=loadmat('Output_u.mat')['Output_u']
        y_train_3=loadmat('Output_v.mat')['Output_v']
        y_train_4=loadmat('Output_p.mat')['Output_p']
        

        x_train_1 = np.expand_dims(x_train_1,axis=3)
        x_train_2 = np.expand_dims(x_train_2,axis=2)
        
        
        return x_train_1,x_train_2,y_train_1,y_train_2,y_train_3,y_train_4


def build_model(num_conv,num_dense,inp_optim,inp_loss):


        x_train_1=loadmat('SDF_values.mat')['SDF_values']
        x_train_2=loadmat('input_re_alpha.mat')['input_re_alpha']    #Load training data
        y_train_1=loadmat('Output_values.mat')['Output_values']

        y_train_2=loadmat('Output_u.mat')['Output_u']
        y_train_3=loadmat('Output_v.mat')['Output_v']
        y_train_4=loadmat('Output_p.mat')['Output_p']


        x_train_1 = np.expand_dims(x_train_1,axis=3)
        x_train_2 = np.expand_dims(x_train_2,axis=2)

        input1=Input(shape=(150,150,1),name='input1')
        input2=Input(shape=(2,1),name='input2')

        num_units=np.zeros(num_conv,dtype=int)
        num_kernel=np.zeros(num_conv,dtype=int)
        num_strides=np.zeros(num_conv,dtype=int)


        #------------------------------------------------------------------------------------------------------




        with open('config_file.csv','r') as csvfile:
            info = csv.reader(csvfile, delimiter=',')
            info=list(info)


        user_input_channels=int(info[0][1])

        #--------------------------------------------------------------------------------------------------------

	

        print("Constructing the convolutional layers:")    
 #----------------------------------------------------------------------------------Convolutional Layers Construction  
        maxpling=[]
        maxpling_size=[]
        for i in range(1,num_conv+1):
            print("Convolutional Layer %d"%(i))
            inp_numfilters=int(info[i][1])
            num_units[i-1]=int(info[i][1])
            inp_shape=int(info[i][2])
            num_kernel[i-1]=int(info[i][2])
            inp_stride=int(info[i][3])
            num_strides[i-1]=int(info[i][3])
            inp_activation=info[i][4]
            
            if i==1:
		
                output=Conv2D(inp_numfilters,(inp_shape,inp_shape),strides=(inp_stride,inp_stride),activation=inp_activation,kernel_regularizer=regularizers.l2(0.00001))(input1)
                output=BatchNormalization()(output)
            else:
                output=Conv2D(inp_numfilters,(inp_shape,inp_shape),strides=(inp_stride,inp_stride),activation=inp_activation,kernel_regularizer=regularizers.l2(0.00001))(output)
                output=BatchNormalization()(output)
            inp_maxpling=info[i][5]
            inp_maxpling_size=int(info[i][6])
            maxpling.append(inp_maxpling)
            maxpling_size.append(inp_maxpling_size)
            if inp_maxpling=='Y':
                output=MaxPooling2D(pool_size=(inp_maxpling_size,inp_maxpling_size),strides=(inp_maxpling_size,inp_maxpling_size))(output)
                
            else:
                continue
                
 #------------------------------------------------------------------------------------       
        shape_2=output._keras_shape
        print(shape_2)
        output=Reshape((-1,1))(output)
        output=concatenate([output,input2],axis=-2)
        output=Flatten()(output)
        shape_1=output._keras_shape
        print(shape_1)
        
        print("Constructing the Dense Layers")
 #------------------------------------------------------------------------------------Dense Layers Construction       
        for i in range(1,num_dense+1): 
            print("Dense Layer %d"%(i))
            inp_numunits=int(info[num_conv+i][1])
            inp_activation=info[num_conv+i][2]
            
            if i==num_dense:
                output=Dense(shape_1[1]-2,activation=inp_activation,kernel_regularizer=regularizers.l2(0.00001))(output)
                output=BatchNormalization()(output)
            else:
                output=Dense(inp_numunits,activation=inp_activation,kernel_regularizer=regularizers.l2(0.00001))(output)
                output=BatchNormalization()(output)
        
            
 #------------------------------------------------------------------------------------
        print(output._keras_shape)
        output=Reshape((shape_2[1],shape_2[2],shape_2[3]))(output)
        print("Creating correspondingly symmetrical Deconvolutional layers")
        
        if user_input_channels==3:
 #------------------------------------------------------------------------------------ DeConvolutional Layers Construction 
    
            for i in range(1,num_conv+1):
                if maxpling[num_conv-i]=='Y':
                    output=UpSampling2D(size=(maxpling_size[num_conv-i],maxpling_size[num_conv-i]))(output)
                if i==num_conv:
                    output=Conv2DTranspose(3,(num_kernel[num_conv-i],num_kernel[num_conv-i]),strides=(num_strides[num_conv-i],num_strides[num_conv-i]),activation='linear',kernel_regularizer=regularizers.l2(0.00001))(output)
                else:
                    output=Conv2DTranspose(num_units[num_conv-i],(num_kernel[num_conv-i],num_kernel[num_conv-i]),strides=(num_strides[num_conv-i],num_strides[num_conv-i]),activation=inp_activation,kernel_regularizer=regularizers.l2(0.00001))(output)
                    output=BatchNormalization()(output)
                    
#--------------------------------------------------------------------------------------------------------------------------------  
            model=Model(inputs=[input1,input2],outputs=[output])


 #--------------------------------------------------------------------------------------------------------------------------------  
 #---------------------------------------------------------------------------------------------------------------------   
 #-----------------------------------------------------------------------------------------------------------------------------
        else:
        
            output1=output
            output2=output
            output3=output
     #-----------------------------------------------------------------------------------3-path Deconvolution output       
            for i in range(1,num_conv+1):
                if maxpling[num_conv-i]=='Y':
                    output1=UpSampling2D(size=(maxpling_size[num_conv-i],maxpling_size[num_conv-i]))(output1)
                if i==num_conv:
                    output1=Conv2DTranspose(1,(num_kernel[num_conv-i],num_kernel[num_conv-i]),strides=(num_strides[num_conv-i],num_strides[num_conv-i]),activation='linear',kernel_regularizer=regularizers.l2(0.00001))(output1)

                elif i==1:
                    output1=Conv2DTranspose(num_units[num_conv-i],(num_kernel[num_conv-i],num_kernel[num_conv-i]),strides=(num_strides[num_conv-i],num_strides[num_conv-i]),activation=inp_activation,kernel_regularizer=regularizers.l2(0.00001))(output1)
                    output1=BatchNormalization()(output1)
                else:
                    output1=Conv2DTranspose(num_units[num_conv-i],(num_kernel[num_conv-i],num_kernel[num_conv-i]),strides=(num_strides[num_conv-i],num_strides[num_conv-i]),activation=inp_activation,kernel_regularizer=regularizers.l2(0.00001))(output1)
                    output1=BatchNormalization()(output1)
            for i in range(1,num_conv+1):
                if maxpling[num_conv-i]=='Y':
                    output2=UpSampling2D(size=(maxpling_size[num_conv-i],maxpling_size[num_conv-i]))(output2)
                if i==num_conv:
                    output2=Conv2DTranspose(1,(num_kernel[num_conv-i],num_kernel[num_conv-i]),strides=(num_strides[num_conv-i],num_strides[num_conv-i]),activation='linear',kernel_regularizer=regularizers.l2(0.00001))(output2)
                elif i==1:
                    output2=Conv2DTranspose(num_units[num_conv-i],(num_kernel[num_conv-i],num_kernel[num_conv-i]),strides=(num_strides[num_conv-i],num_strides[num_conv-i]),activation=inp_activation,kernel_regularizer=regularizers.l2(0.00001))(output2)
                    output2=BatchNormalization()(output2)
                else:
                    output2=Conv2DTranspose(num_units[num_conv-i],(num_kernel[num_conv-i],num_kernel[num_conv-i]),strides=(num_strides[num_conv-i],num_strides[num_conv-i]),activation=inp_activation,kernel_regularizer=regularizers.l2(0.00001))(output2)
                    output2=BatchNormalization()(output2)
            for i in range(1,num_conv+1):
                if maxpling[num_conv-i]=='Y':
                    output3=UpSampling2D(size=(maxpling_size[num_conv-i],maxpling_size[num_conv-i]))(output3)
                if i==num_conv:
                    output3=Conv2DTranspose(1,(num_kernel[num_conv-i],num_kernel[num_conv-i]),strides=(num_strides[num_conv-i],num_strides[num_conv-i]),activation='linear',kernel_regularizer=regularizers.l2(0.00001))(output3)
                elif i==1:
                    output3=Conv2DTranspose(num_units[num_conv-i],(num_kernel[num_conv-i],num_kernel[num_conv-i]),strides=(num_strides[num_conv-i],num_strides[num_conv-i]),activation=inp_activation,kernel_regularizer=regularizers.l2(0.00001))(output3)
                    output3=BatchNormalization()(output3)
                else:
                    output3=Conv2DTranspose(num_units[num_conv-i],(num_kernel[num_conv-i],num_kernel[num_conv-i]),strides=(num_strides[num_conv-i],num_strides[num_conv-i]),activation=inp_activation,kernel_regularizer=regularizers.l2(0.00001))(output3)
                    output3=BatchNormalization()(output3)
            model=Model(inputs=[input1,input2],outputs=[output1,output2,output3])  
     #-----------------------------------------------------------------------------------------------------------------------------
     #-----------------------------------------------------------------------------------------------------------------------------





        model.compile(optimizer=inp_optim,loss='mse')

        model_json = model.to_json()
        with open("model.json", "w") as json_file:
             json_file.write(model_json)

        model.summary()
        return model
    


def train_model(batch_sz,eps,val_splt,model):
    with open('config_file_2.csv','r') as csvfile:
        info=csv.reader(csvfile,delimiter=',')
        info=list(info)
    exp_no=int(info[0][0])
    user_input_channels=int(info[0][1])
    user_inp=info[0][2]
    if user_inp=='Y':
        inp_delta=float(info[0][3])
        inp_mineps=float(info[0][4])
    
    x_train_1=loadmat('SDF_values.mat')['SDF_values']
    x_train_2=loadmat('input_re_alpha.mat')['input_re_alpha']    #Load training data
    y_train_1=loadmat('Output_values.mat')['Output_values']
    
    y_train_2=loadmat('Output_u.mat')['Output_u']
    y_train_3=loadmat('Output_v.mat')['Output_v']
    y_train_4=loadmat('Output_p.mat')['Output_p']
    '''
    y_train_2=np.expand_dims(y_train_2,axis=3)
    y_train_3=np.expand_dims(y_train_3,axis=3)
    y_train_4=np.expand_dims(y_train_4,axis=3)
    
    x_train_1 = np.expand_dims(x_train_1,axis=3)
    x_train_2 = np.expand_dims(x_train_2,axis=2)
    '''
    
    csv_logger=CSVLogger('training_%d.csv'%(exp_no));
    
    checkpoint = ModelCheckpoint('weights.best.hdf5', monitor='val_loss', verbose=1, save_best_only=True, mode='min')    
    
    if user_inp=='Y':
    
            
            earlystopping=keras.callbacks.EarlyStopping(monitor='mape',min_delta=inp_delta,patience=inp_mineps)
            if user_input_channels==3:
                model.fit([x_train_1,x_train_2],[y_train_1],batch_size=batch_sz,epochs=eps,validation_split=val_splt,callbacks=[csv_logger,earlystopping,checkpoint])
            else:
                model.fit([x_train_1,x_train_2],[y_train_2,y_train_3,y_train_4],batch_size=batch_sz,epochs=eps,validation_split=val_splt,callbacks=[csv_logger,earlystopping,checkpoint])
    else:
            if user_input_channels==3:
                model.fit([x_train_1,x_train_2],[y_train_1],batch_size=batch_sz,epochs=eps,validation_split=val_splt,callbacks=[csv_logger,checkpoint])
            else:
                model.fit([x_train_1,x_train_2],[y_train_2,y_train_3,y_train_4],batch_size=batch_sz,epochs=eps,validation_split=val_splt,callbacks=[csv_logger,checkpoint])    
    
    model.save('Network_Expt_%d.h5'%(exp_no))
    return model
    
        



def model_prediction(exp_no,model,x1,x2):
    
    
    x_1=loadmat(x1+'.mat')[x1]
    x_2=loadmat(x2+'.mat')[x2]
#    x_1 = np.expand_dims(x_1,axis=3)
#    x_2 = np.expand_dims(x_2,axis=2)
    


    ans=model.predict([x_1,x_2])
    
    Model_output=scipy.io.savemat('prediction_expt_%d.mat'%(exp_no),dict(ans=ans))   # Matlab file check_ans compares the training data and output
  
                                                             #and output on training data
    
    
    
    return ans

def save_model(model,modelname):
    model.save(modelname+'.h5')
    return


def load(modelname):
    model=load_model(modelname)
    return model

