import os
from os import sys, path
import back_end


if __name__=='__main__':

    
    num_conv_layers=3
    num_dense_layers=1
    reg_weight= 1E-5
    num_datapts= 252
    num_training_iters=25
    val_split=0.15
    inp_activation='swish'

    optim_name='adam'
    loss_type='mse'

    model=back_end.build_model(num_conv_layers,num_dense_layers,optim_name,loss_type,reg_weight,inp_activation)


    model=back_end.train_model(num_datapts,num_training_iters,val_split,model)


