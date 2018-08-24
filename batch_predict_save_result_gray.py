import tensorflow as tf
import numpy as np
import os,glob,cv2
import sys,argparse

# process a folder of images
# First, pass the path of the image
#dir_path = os.path.dirname(os.path.realpath(__file__))
#print (dir_path)
#image_path = sys.argv[1]
#image_path = 'testing_data/fcs/'
#filename = dir_path + '/' +image_path
foldername = 'track'
dir_path = './testing_data/' + foldername
save_path = './result/' + foldername
#save_path = './result/gnd2_2'
filename = os.listdir(dir_path)
#print (filename)
image_size = 40   #128
num_channels = 1
images = []
font = cv2.FONT_HERSHEY_SIMPLEX
# Reading the image using OpenCV
for i in filename:
    imagename = dir_path + '/' + i
    # print (imagename)
    image = cv2.imread(imagename, 0)
    #image = cv2.imread('300_1.jpg')   # testing file
    # Resizing the image to our desired size and preprocessing will be done exactly as done during training
    image = cv2.resize(image, (image_size, image_size), 0, 0, cv2.INTER_LINEAR)
    images.append(image)
    #np.append(images, image)
images = np.array(images, dtype=np.uint8)
images = images.astype('float32')
images = np.multiply(images, 1.0/255.0)
#The input to the network is of shape [None image_size image_size num_channels]. Hence we reshape.
x_batch = images.reshape(len(filename), image_size, image_size, num_channels)   #first variable is number of images

## Let us restore the saved model
sess = tf.Session()
# Step-1: Recreate the network graph. At this step only graph is created.
saver = tf.train.import_meta_graph('./model_gray/save_model/fcs-model.meta')   # ./model1/fcs-model.meta
#saver = tf.train.import_meta_graph('model2/fcs-model2.meta')
# Step-2: Now let's load the weights saved using the restore method.
saver.restore(sess, tf.train.latest_checkpoint('./model_gray/save_model/'))   #./model1/
#saver.restore(sess, tf.train.latest_checkpoint('./model2/'))
# Accessing the default graph which we have restored
graph = tf.get_default_graph()

# Now, let's get hold of the op that we can be processed to get the output.
# In the original network y_pred is the tensor that is the prediction of the network
y_pred = graph.get_tensor_by_name("y_pred:0")

## Let's feed the images to the input placeholders
x= graph.get_tensor_by_name("x:0")
y_true = graph.get_tensor_by_name("y_true:0")
y_test_images = np.zeros((1, 3))


### Creating the feed_dict that is required to be fed to calculate y_pred
feed_dict_testing = {x: x_batch, y_true: y_test_images}
result = sess.run(y_pred, feed_dict=feed_dict_testing)
# result is of this format [probabiliy_of_rose probability_of_sunflower]
#print(result)
ctr = 0
fontScale = 0.5
thickness = 1
def putresult(img, r):
    #r = r.astype(int)
    height, width, _ = img.shape
    if r >= 0.5:
        green = 255
        red = 255 * 2 * (1-r)
    else:
        green = 255 * 2 * r
        red = 255
    color = (0, green, red)
    r = ("%.3f%%" % (r * 100))
    text = str(r)
    textsize = cv2.getTextSize(text, font, fontScale, thickness)
    text_width = textsize[0][0]
    text_height = textsize[0][1]
    textorg = (int((width - text_width) / 2), height + text_height + 1)   # +1 to show number better
    #print(textorg)
    blank_image = np.zeros((height + textsize[0][1] + 5, width, 3), np.uint8)   # create blank image, +5 to show completed
    blank_image[:] = color   # fill with color
    blank_image[0:height, 0:width] = img   # copy image into it
    cv2.putText(blank_image, text, textorg, font, fontScale, (0, 0, 0), thickness, cv2.LINE_AA)   # put text at bottom
    print(r)
    return blank_image
for j in filename:
    imgname = j
    print(imgname)
    print(result[ctr])
    imgname2 = dir_path + '/' + j
    new_image = cv2.imread(imgname2)
    #new_image = cv2.resize(new_image, (100, 100))   #resize
    new_image = putresult(new_image, result[ctr][2])   # 0, 1, 2 for fcs, ground, track
    cv2.imwrite(save_path + '/' + imgname, new_image)
    ctr += 1

