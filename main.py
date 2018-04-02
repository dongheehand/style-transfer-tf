
# coding: utf-8

# In[ ]:


from vgg19 import *
import cv2
import tensorflow as tf
import numpy as np
import argparse
import transfer


# In[ ]:


parser = argparse.ArgumentParser()

# options for image style transfering
parser.add_argument("--style_image_path",type = str, 
                    help = "File path of style image", default = "input/starry-night.jpg")
parser.add_argument("--content_image_path",type = str, 
                    help = "File path of content image", default = "input/tubingen.jpg")
parser.add_argument("--output_image_path",type = str, 
                    help = "File path of result image", default = "output/result.png")
parser.add_argument("--alpha",type = float, 
                    help = "The coefficient of content loss", default = 0.001)
parser.add_argument("--beta",type = float, 
                    help = "The coefficient of style loss", default = 500.0)
parser.add_argument("--vgg_model_path",type = str,
                    help = "File path of pre-trained vgg19 model", default = "vgg19/vgg19.npy")
parser.add_argument("--content_loss_layer",type = str,
                    help = "Layer(s) included content representation", default = ['conv4_2'])
parser.add_argument("--style_loss_layer",type = str,
                    help = "Layer(s) included style representation", default = ['relu1_1', 'relu2_1', 'relu3_1', 'relu4_1', 'relu5_1'])
parser.add_argument("--height",type = int,
                    help = "The height of output image", default = 384)
parser.add_argument("--width",type = int,
                    help = "The width of output image", default = 512)
parser.add_argument("--epoch",type = int,
                    help = "The number of epoch", default = 2000)

args = parser.parse_args()

style = cv2.cvtColor(cv2.imread(args.style_image_path),cv2.COLOR_BGR2RGB)
content = cv2.cvtColor(cv2.imread(args.content_image_path),cv2.COLOR_BGR2RGB)
output_path = args.output_image_path
alpha = args.alpha
beta = args.beta
vgg_path = args.vgg_model_path
content_loss = args.content_loss_layer
style_loss = args.style_loss_layer
height = args.height
width = args.width
epoch = args.epoch
c = 3

style = cv2.resize(style,(width,height),interpolation = cv2.INTER_LANCZOS4)
content = cv2.resize(content,(width,height),interpolation = cv2.INTER_LANCZOS4)

style = np.float32(style.reshape(1,height,width,c))
content = np.float32(content.reshape(1,height,width,c))
vgg_const = Vgg19(vgg_path)
vgg_var = Vgg19(vgg_path)


# In[ ]:


image_transfer = transfer.image_transfer(vgg_const = vgg_const, vgg_var = vgg_var,
                                        style_loss_layer = style_loss , content_loss_layer = content_loss,
                                        style_img = style, content_img = content, alpha = alpha, beta = beta,
                                        height = height , width = width, channel = c, max_epoch = epoch)
loss, opt = image_transfer()


# In[ ]:


config = tf.ConfigProto()
config.gpu_options.allow_growth=True
sess = tf.Session(config=config)
sess.run(tf.global_variables_initializer())


# In[ ]:


opt.minimize(sess)
output = sess.run(image_transfer.output_img)
output = output[0]
cliped_output = np.clip(output,0.0,255.0)
cv2.imwrite(output_path, cv2.cvtColor(cliped_output,cv2.COLOR_RGB2BGR))

