from keras import backend as K
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
from keras.models import model_from_json
import csv
    


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


def load_training_data():

        
        x_train_1=loadmat('SDF_values.mat')['SDF_values']
        x_train_2=loadmat('input_re_alpha.mat')['input_re_alpha']    #Load training data
        y_train_1=loadmat('Output_values.mat')['Output_values']

        y_train_2=loadmat('Output_u.mat')['Output_u']
        y_train_3=loadmat('Output_v.mat')['Output_v']
        y_train_4=loadmat('Output_p.mat')['Output_p']


        x_train_1 = np.expand_dims(x_train_1,axis=3)
        x_train_2 = np.expand_dims(x_train_2,axis=2)

        return x_train_1,x_train_2,y_train_1,y_train_2,y_train_3,y_train_4

def build_conv_layers(input1,num_conv,num_kernel,num_strides,num_units,info,inp_activation,reg_weight):
    print("Constructing the convolutional layers:")    
    maxpling=[]
    maxpling_size=[]
    for i in range(1,num_conv+1):
        print("Setting up Convolutional Layer %d"%(i))
        
        if i==1:
            
            output=Conv2D(num_units[i-1],(num_kernel[i-1],num_kernel[i-1]),strides=(num_strides[i-1],num_strides[i-1]),activation=inp_activation,kernel_regularizer=regularizers.l2(reg_weight))(input1)
            output=BatchNormalization()(output)
        else:
            output=Conv2D(num_units[i-1],(num_kernel[i-1],num_kernel[i-1]),strides=(num_strides[i-1],num_strides[i-1]),activation=inp_activation,kernel_regularizer=regularizers.l2(reg_weight))(output)
            output=BatchNormalization()(output)

        # read whether to use max pooling, and pooling size
        inp_maxpling=info[i][5]
        inp_maxpling_size=int(info[i][6])
        maxpling.append(inp_maxpling)
        maxpling_size.append(inp_maxpling_size)
        if inp_maxpling=='Y':
            output=MaxPooling2D(pool_size=(inp_maxpling_size,inp_maxpling_size),strides=(inp_maxpling_size,inp_maxpling_size))(output)
            
        else:
            continue

    return maxpling,maxpling_size,output

def build_hidden_layers(output,num_conv,num_dense,info,shape_conv_inp2,reg_weight,inp_activation):

    print("Constructing the Dense Layers")
    for i in range(1,num_dense+1): 
        print("Dense Layer %d"%(i))
        inp_numunits=int(info[num_conv+i][1])
        
        # if last dense layer, number of output units is pre determined
        if i==num_dense:
            
            output=Dense(shape_conv_inp2[1]-2,activation=inp_activation,kernel_regularizer=regularizers.l2(reg_weight))(output)
            output=BatchNormalization()(output)

        else:
            output=Dense(inp_numunits,activation=inp_activation,kernel_regularizer=regularizers.l2(reg_weight))(output)
            output=BatchNormalization()(output)

    return output
    
def build_1_path_deconv(output,num_conv,maxpling,maxpling_size,reg_weight,num_kernel,num_strides,num_units,inp_activation):

    print("Creating correspondingly symmetrical Deconvolutional layers (1 path)")
    for i in range(1,num_conv+1):
        if maxpling[num_conv-i]=='Y':
            output=UpSampling2D(size=(maxpling_size[num_conv-i],maxpling_size[num_conv-i]))(output)
        if i==num_conv:
            output=Conv2DTranspose(3,(num_kernel[num_conv-i],num_kernel[num_conv-i]),strides=(num_strides[num_conv-i],num_strides[num_conv-i]),activation='linear',kernel_regularizer=regularizers.l2(reg_weight))(output)
        else:
            output=Conv2DTranspose(num_units[num_conv-i],(num_kernel[num_conv-i],num_kernel[num_conv-i]),strides=(num_strides[num_conv-i],num_strides[num_conv-i]),activation=inp_activation,kernel_regularizer=regularizers.l2(reg_weight))(output)
            output=BatchNormalization()(output)


    outputs_list=[output]

    return outputs_list

def build_3_path_deconv(output,num_conv,maxpling,maxpling_size,reg_weight,num_kernel,num_strides,num_units,inp_activation):
    print("Creating correspondingly symmetrical Deconvolutional layers (3 path)")

    output1=output
    output2=output
    output3=output

    #all_outputs=[output,output,output]

    for i in range(1,num_conv+1):
        if maxpling[num_conv-i]=='Y':
            output1=UpSampling2D(size=(maxpling_size[num_conv-i],maxpling_size[num_conv-i]))(output1)
            output2=UpSampling2D(size=(maxpling_size[num_conv-i],maxpling_size[num_conv-i]))(output2)
            output3=UpSampling2D(size=(maxpling_size[num_conv-i],maxpling_size[num_conv-i]))(output3)
        if i==num_conv:
            output1=Conv2DTranspose(1,(num_kernel[num_conv-i],num_kernel[num_conv-i]),strides=(num_strides[num_conv-i],num_strides[num_conv-i]),activation='linear',kernel_regularizer=regularizers.l2(reg_weight))(output1)
            output2=Conv2DTranspose(1,(num_kernel[num_conv-i],num_kernel[num_conv-i]),strides=(num_strides[num_conv-i],num_strides[num_conv-i]),activation='linear',kernel_regularizer=regularizers.l2(reg_weight))(output2)
            output3=Conv2DTranspose(1,(num_kernel[num_conv-i],num_kernel[num_conv-i]),strides=(num_strides[num_conv-i],num_strides[num_conv-i]),activation='linear',kernel_regularizer=regularizers.l2(reg_weight))(output3)

        elif i==1:
            output1=Conv2DTranspose(num_units[num_conv-i],(num_kernel[num_conv-i],num_kernel[num_conv-i]),strides=(num_strides[num_conv-i],num_strides[num_conv-i]),activation=inp_activation,kernel_regularizer=regularizers.l2(reg_weight))(output1)
            output1=BatchNormalization()(output1)

            output2=Conv2DTranspose(num_units[num_conv-i],(num_kernel[num_conv-i],num_kernel[num_conv-i]),strides=(num_strides[num_conv-i],num_strides[num_conv-i]),activation=inp_activation,kernel_regularizer=regularizers.l2(reg_weight))(output2)
            output2=BatchNormalization()(output2)

            output3=Conv2DTranspose(num_units[num_conv-i],(num_kernel[num_conv-i],num_kernel[num_conv-i]),strides=(num_strides[num_conv-i],num_strides[num_conv-i]),activation=inp_activation,kernel_regularizer=regularizers.l2(reg_weight))(output3)
            output3=BatchNormalization()(output3)
        else:
            output1=Conv2DTranspose(num_units[num_conv-i],(num_kernel[num_conv-i],num_kernel[num_conv-i]),strides=(num_strides[num_conv-i],num_strides[num_conv-i]),activation=inp_activation,kernel_regularizer=regularizers.l2(reg_weight))(output1)
            output1=BatchNormalization()(output1)

            output2=Conv2DTranspose(num_units[num_conv-i],(num_kernel[num_conv-i],num_kernel[num_conv-i]),strides=(num_strides[num_conv-i],num_strides[num_conv-i]),activation=inp_activation,kernel_regularizer=regularizers.l2(reg_weight))(output2)
            output2=BatchNormalization()(output2)

            output3=Conv2DTranspose(num_units[num_conv-i],(num_kernel[num_conv-i],num_kernel[num_conv-i]),strides=(num_strides[num_conv-i],num_strides[num_conv-i]),activation=inp_activation,kernel_regularizer=regularizers.l2(reg_weight))(output3)
            output3=BatchNormalization()(output3)

    outputs_list=[output1,output2,output3]

    return outputs_list


def build_model(num_conv,num_dense,inp_optim,inp_loss,reg_weight,inp_activation):



        # define input layer
        # SDF input

        #TODO remove hardcoded shapes from here
        input1=Input(shape=(150,150,1),name='input1')
        # vector of scalars
        input2=Input(shape=(2,1),name='input2')

        # initialize
        num_units=np.zeros(num_conv,dtype=int)
        num_kernel=np.zeros(num_conv,dtype=int)
        num_strides=np.zeros(num_conv,dtype=int)

        # read user inputs
        with open('config_file.csv','r') as csvfile:
            info = csv.reader(csvfile, delimiter=',')
            info=list(info)

        for i in range(1,num_conv+1):
            print("Setting up Convolutional Layer %d"%(i))
            inp_numfilters=int(info[i][1])
            num_units[i-1]=int(info[i][1])
            inp_shape=int(info[i][2])
            num_kernel[i-1]=int(info[i][2])
            inp_stride=int(info[i][3])
            num_strides[i-1]=int(info[i][3])

        deconv_channels=int(info[0][1])

        # construct convolutional layers 
        maxpling,maxpling_size,output = build_conv_layers(input1,num_conv,num_kernel,num_strides,num_units,info,inp_activation,reg_weight) 
    
        # shape of output of convolutions
        shape_conv_out=output.shape # this shape will be re used later

        # reshape to vector and concatenate second input vector
        output=Reshape((-1,1))(output)
        output=concatenate([output,input2],axis=-2)
        output=Flatten()(output)
        shape_conv_inp2=output.shape
       
        # construct dense layers
        output=build_hidden_layers(output,num_conv,num_dense,info,shape_conv_inp2,reg_weight,inp_activation)
           
        # reshape output of dense layers for deconvolution
        output=Reshape((shape_conv_out[1],shape_conv_out[2],shape_conv_out[3]))(output)
       
        #if user selects 3-channel (single path) output
        if deconv_channels==3:
            outputs_list=build_1_path_deconv(output,num_conv,maxpling,maxpling_size,reg_weight,num_kernel,num_strides,num_units,inp_activation)

        else:
        
            outputs_list=build_3_path_deconv(output,num_conv,maxpling,maxpling_size,reg_weight,num_kernel,num_strides,num_units,inp_activation)

        # compile model
        model=Model(inputs=[input1,input2],outputs=outputs_list)

        model.compile(optimizer=inp_optim,loss=inp_loss)

        # save model to string
        model_json = model.to_json()
        with open("model.json", "w") as json_file:
             json_file.write(model_json)
        # print model statistics
        model.summary()
        return model
    


def train_model(batch_sz,eps,val_splt,model):

    with open('training_config_file.csv','r') as csvfile:
        info=csv.reader(csvfile,delimiter=',')
        info=list(info)

    exp_no=int(info[0][0])
    user_input_channels=int(info[0][1])
    user_inp=info[0][2]

    inp_delta=float(info[0][3])
    inp_mineps=float(info[0][4])
    
    x_train_1,x_train_2,y_train_1,y_train_2,y_train_3,y_train_4=load_training_data()
    
    csv_logger=CSVLogger('training_%d.csv'%(exp_no));
    
    checkpoint = ModelCheckpoint('weights.best.keras', monitor='val_loss', verbose=1, save_best_only=True, mode='min')    
    

    earlystopping=keras.callbacks.EarlyStopping(monitor='mape',min_delta=inp_delta,patience=inp_mineps,mode='min')
    if user_input_channels==3:
        model.fit([x_train_1,x_train_2],[y_train_1],batch_size=batch_sz,epochs=eps,validation_split=val_splt,callbacks=[csv_logger,earlystopping,checkpoint])
    else:
        model.fit([x_train_1,x_train_2],[y_train_2,y_train_3,y_train_4],batch_size=batch_sz,epochs=eps,validation_split=val_splt,callbacks=[csv_logger,earlystopping,checkpoint])
    
    model.save('Network_Expt_%d.h5'%(exp_no))
    return model
    

def model_prediction(exp_no,model,x1,x2):
    
    x_1=loadmat(x1+'.mat')[x1]
    x_2=loadmat(x2+'.mat')[x2]

    ans=model.predict([x_1,x_2])
    
    Model_output=scipy.io.savemat('prediction_expt_%d.mat'%(exp_no),dict(ans=ans))   # Matlab file check_ans compares the training data and output
  
    return ans

def save_model(model,modelname):
    model.save(modelname+'.h5')
    return


def load(modelname):
    model=load_model(modelname)
    return model

