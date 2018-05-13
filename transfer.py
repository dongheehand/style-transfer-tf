import tensorflow as tf
from vgg19 import *
import numpy as np

class style_transfer_net():
    
    def __init__(self, conf):
        
        self.height = conf['height']        
        self.width = conf['width']
        self.epoch = conf['epoch']
        self.vgg_model_path = conf['vgg_model_path']
        self.style_loss_layer = conf['style_loss_layer']
        self.content_loss_layer = conf['content_loss_layer']
        self.channel = conf['channel']
        self.alpha = conf['alpha']
        self.beta = conf['beta']
    
    def build_graph(self):
        
        self.style_img = tf.placeholder(name = 'style_img', shape = (1, self.height, self.width, self.channel), dtype = tf.float32)
        self.content_img = tf.placeholder(name = 'content_img', shape = (1, self.height, self.width, self.channel), dtype = tf.float32)
        self.output_img = tf.Variable(self.content_img, trainable = True, dtype = tf.float32)
        
        vgg_style = Vgg19(self.vgg_model_path)
        vgg_content = Vgg19(self.vgg_model_path)
        vgg_output = Vgg19(self.vgg_model_path)
        
        vgg_style.build(self.style_img)
        vgg_content.build(self.content_img)
        vgg_output.build(self.output_img)
        
        target_style_gram = self.get_style_gram_matrix(vgg_style, self.style_loss_layer)
        output_style_gram = self.get_style_gram_matrix(vgg_output, self.style_loss_layer)
        
        target_content_vec = self.get_content_vec(vgg_content, self.content_loss_layer)
        output_content_vec = self.get_content_vec(vgg_output, self.content_loss_layer)
        
        self.style_loss = self.get_style_loss(target_style_gram, output_style_gram)
        self.content_loss = self.get_content_loss(target_content_vec, output_content_vec)
        
        self.loss = self.alpha * self.content_loss + self.beta * self.style_loss
        
        self.opt = tf.contrib.opt.ScipyOptimizerInterface(self.loss, var_list = [self.output_img], method = 'L-BFGS-B', options = {'maxiter':self.epoch})
        
    def get_content_vec(self, vgg_network, conv_layer):
        
        return_array = np.array(())
        
        for ele in conv_layer:
            return_array = np.append(return_array,getattr(vgg_network,ele))
        
        return return_array

    def get_style_gram_matrix(self, vgg_network, conv_layer):
        
        return_array = np.array(())

        for ele in conv_layer:
            feature = getattr(vgg_network,ele) 
            size = tf.shape(feature)
            return_array = np.append(return_array, self.Gram_matrix(feature,size))

        return return_array

    def Gram_matrix(self, feature_map_tensor,size):

        reshaped_feature_map = tf.reshape(feature_map_tensor, (size[0], size[1] * size[2], size[3]))
        normalization = 2.0 * tf.cast(size[1] * size[2] * size[3] , tf.float32)
        return tf.div(tf.matmul(tf.transpose(reshaped_feature_map, perm = [0,2,1]),reshaped_feature_map) ,normalization)

    
    def get_style_loss(self, style_vec, transfer_vec):

        loss = tf.constant(value = 0.0 , dtype = tf.float32)

        for i in range(len(style_vec)):
            _loss = tf.nn.l2_loss(style_vec[i] - transfer_vec[i])
            loss += 0.2 * _loss

        return loss
    
    def get_content_loss(self, content_vec, transfer_vec):

        loss = tf.constant(value = 0.0 , dtype = tf.float32)
        
        for i in range(len(content_vec)):
            _loss = tf.nn.l2_loss(content_vec[i] - transfer_vec[i])
            loss += _loss

        return 0.5 * loss        

