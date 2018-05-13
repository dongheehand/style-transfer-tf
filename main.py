import tensorflow as tf
import numpy as np
import argparse
import transfer
from PIL import Image
from transfer import style_transfer_net

parser = argparse.ArgumentParser()

# options for image style transfering
parser.add_argument("--style_image_path",type = str, 
                    help = "File path of style image", default = "input/starry-night.jpg")
parser.add_argument("--content_image_path",type = str, 
                    help = "File path of content image", default = "input/tubingen.jpg")
parser.add_argument("--output_image_path",type = str, 
                    help = "File path of result image", default = "output/result.png")
parser.add_argument("--height",type = int,
                    help = "The height of output image", default = 384)
parser.add_argument("--width",type = int,
                    help = "The width of output image", default = 512)
parser.add_argument("--vgg_model_path",type = str,
                    help = "File path of pre-trained vgg19 model", default = "vgg19/vgg19.npy")
parser.add_argument("--style_loss_layer",type = str,
                    help = "Layer(s) included style representation", default = ['relu1_1', 'relu2_1', 'relu3_1', 'relu4_1', 'relu5_1'])
parser.add_argument("--content_loss_layer",type = str,
                    help = "Layer(s) included content representation", default = ['conv4_2'])
parser.add_argument("--alpha",type = float, 
                    help = "The coefficient of content loss", default = 0.001)
parser.add_argument("--beta",type = float, 
                    help = "The coefficient of style loss", default = 10.0)
parser.add_argument("--epoch",type = int,
                    help = "The number of epoch", default = 2000)

args = parser.parse_args()

conf = {}
conf['style_image_path'] = args.style_image_path
conf['content_image_path'] = args.content_image_path
conf['output_image_path'] = args.output_image_path
conf['height'] = args.height
conf['width'] = args.width
conf['vgg_model_path'] = args.vgg_model_path
conf['style_loss_layer'] = args.style_loss_layer
conf['content_loss_layer'] = args.content_loss_layer
conf['alpha'] = args.alpha
conf['beta'] = args.beta
conf['epoch'] = args.epoch
conf['channel'] = 3

style_image = Image.open(conf['style_image_path'])
content_image = Image.open(conf['content_image_path'])

style_image = style_image.resize((conf['width'], conf['height']), Image.LANCZOS)
content_image = content_image.resize((conf['width'],conf['height']), Image.LANCZOS)

style_image = np.expand_dims(np.array(style_image), axis = 0)
content_image = np.expand_dims(np.array(content_image), axis = 0)

transfer_net = style_transfer_net(conf)
transfer_net.build_graph()

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config = config)
sess.run(tf.global_variables_initializer(), feed_dict = {transfer_net.content_img : content_image})

transfer_net.opt.minimize(sess, feed_dict = {transfer_net.style_img : style_image, transfer_net.content_img : content_image})

output = sess.run(transfer_net.output_img)
output = np.clip(output[0], 0, 255)
output = output.astype(np.uint8)
output = Image.fromarray(output)
output.save(conf['output_image_path'])

