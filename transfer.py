
# coding: utf-8

# In[ ]:


import tensorflow as tf
from vgg19 import *
import numpy as np


# In[ ]:


class image_transfer():
    
    def __init__(self, vgg_const, vgg_var, style_loss_layer, content_loss_layer, 
                 style_img, content_img, alpha, beta, height, width, channel, max_epoch):
        
        self.vgg_const = vgg_const
        self.vgg_var = vgg_var
        self.style_layer = style_loss_layer
        self.content_layer = content_loss_layer
        self.alpha = alpha
        self.beta = beta
        self.height = height
        self.width = width
        self.channel = channel
        self.max_epoch = max_epoch
        
        ## images are [1,height, width, channel] and each value is 0~255

        self.style_img = tf.constant(value = style_img, dtype = tf.float32, shape = [1,self.height,self.width,self.channel], name = 'style_img')
        self.content_img = tf.constant(value = content_img, dtype = tf.float32, shape = [1,self.height,self.width,self.channel], name = 'content_img')
        self.output_img = tf.Variable(content_img, trainable = True, dtype = tf.float32)
 
        self.images = tf.concat([self.style_img, self.content_img], axis = 0)
        
        self.vgg_const.build(self.images)
        self.vgg_var.build(self.output_img)
        
        self.content_vecs = self.get_content_vec(self.vgg_const, self.content_layer)
        self.coeff_list , self.style_gram_matrix = self.get_style_gram_matrix(self.vgg_const, self.style_layer)
        
    def __call__(self):
        
        content_var_vecs = self.get_content_vec(self.vgg_var, self.content_layer, is_trainable = True)
        _, style_var_gram_matrix = self.get_style_gram_matrix(self.vgg_var, self.style_layer, is_trainable = True)
        
        content_loss = tf.constant(0.0,dtype=tf.float32)
        style_loss = tf.constant(0.0,dtype=tf.float32)
        
        for i in range(len(content_var_vecs)):
            content_loss += tf.reduce_sum((content_var_vecs[i]-self.content_vecs[i])**2)/2
            
        for i in range(len(style_var_gram_matrix)):
            style_loss += 0.2 * (tf.nn.l2_loss((style_var_gram_matrix[i]-self.style_gram_matrix[i])/tf.cast(x=self.coeff_list[i],dtype=tf.float32)))/2.0
        
        self.loss = self.alpha * content_loss + self.beta * style_loss
        
        self.opt = tf.contrib.opt.ScipyOptimizerInterface(self.loss, var_list = [self.output_img], method = 'L-BFGS-B', options = {'maxiter':self.max_epoch})

        return self.loss, self.opt
    
    def get_content_vec(self, vgg_network, conv_layer, is_trainable = False):
        return_array = np.array(())
        for ele in conv_layer:
            if is_trainable is True:
                return_array = np.append(return_array,getattr(vgg_network,ele)[0])
            else:
                return_array = np.append(return_array,getattr(vgg_network,ele)[1])
                
        return return_array
    
    def get_style_gram_matrix(self, vgg_network, conv_layer, is_trainable = False):
        return_array = np.array(())
        coeff_array = np.array(())      ## list of (N_l * M_l)
        for ele in conv_layer:
            if is_trainable is True :
                feature = (getattr(vgg_network,ele)[0])
                size = tf.shape(feature)
                coeff_array = np.append(coeff_array,np.array((size[0]*size[1]*size[2])))
                return_array = np.append(return_array,self.Gram_matrix(feature,size[:3]))
                
            else:
                feature = (getattr(vgg_network,ele)[0]) 
                size = tf.shape(feature)
                coeff_array = np.append(coeff_array,np.array((size[0]*size[1]*size[2])))
                return_array = np.append(return_array,self.Gram_matrix(feature,size[:3]))
                
        coeff_array.reshape(len(coeff_array),1)
        return coeff_array, return_array

    def Gram_matrix(self,feature_map_tensor,size):
        reshaped_feature_map = tf.reshape(feature_map_tensor, (size[0]*size[1],size[2]))
        return tf.matmul(tf.transpose(reshaped_feature_map),reshaped_feature_map)
    

