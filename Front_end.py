
# coding: utf-8

# In[1]:


import os
from os import sys, path
sys.path.append(r'/home/saakaar/Desktop/Neural Networks/All_data')


# In[2]:


import back_end


# In[3]:


model=back_end.build_model(3,1,'adam','mse')


# In[4]:


model=back_end.train_model(252,25000,0.15,model)


