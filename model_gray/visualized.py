import dataset_gray
import cv2
import tensorflow as tf
import time
from datetime import timedelta
import math
import random
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.axes_rgb import make_rgb_axes, RGBAxes
from sklearn import preprocessing

# Adding Seed so that random initialization is consistent
from numpy.random import seed

seed(1)
from tensorflow import set_random_seed

set_random_seed(2)

batch_size = 3500   # 32

# Prepare input data
classes = ['fcs', 'ground', 'track']
num_classes = len(classes)

# 20% of the data will automatically be used for validation
validation_size = 0.2
img_size = 40  #
num_channels = 1   # first conv layer input channel
train_path = '../training_data'

# We shall load all the training and validation images and labels into memory using openCV and use that during training
'''data = dataset_grey.read_train_sets(train_path, img_size, classes, validation_size=validation_size)

print("Complete reading input data. Will Now print a snippet of it")
print("Number of files in Training-set:\t\t{}".format(len(data.train.labels)))
print("Number of files in Validation-set:\t{}".format(len(data.valid.labels)))'''

session = tf.Session()
x = tf.placeholder(tf.float32, shape=[None, img_size, img_size, num_channels], name='x')   # num_channels

## labels
y_true = tf.placeholder(tf.float32, shape=[None, num_classes], name='y_true')
y_true_cls = tf.argmax(y_true, dimension=1)

##Network graph params
filter_size_conv1 = 3
num_filters_conv1 = 64   # max_pooling

filter_size_conv2 = 1
num_filters_conv2 = 32

filter_size_conv3 = 3
num_filters_conv3 = 32   # max_pooling

filter_size_conv4 = 3
num_filters_conv4 = 64   # max_pooling

fc_layer_size1 = 256
fc_layer_size2 = 256
fc_layer_size3 = 128


def create_weights(shape, weight_loss=None):
    weights = tf.Variable(tf.truncated_normal(shape, stddev=0.05))
    if weight_loss is not None:   # L2 Regularization
        wl = tf.multiply(tf.nn.l2_loss(weights), weight_loss, name='weight_loss')
        tf.add_to_collection('losses', wl)
    return weights


def create_biases(size):
    return tf.Variable(tf.constant(0.05, shape=[size]))


def create_convolutional_layer(input,
                               num_input_channels,
                               conv_filter_size,
                               num_filters,
                               max_pooling=True):
    ## We shall define the weights that will be trained using create_weights function.
    weights = create_weights(shape=[conv_filter_size, conv_filter_size, num_input_channels, num_filters])
    ## We create biases using the create_biases function. These are also trained.
    biases = create_biases(num_filters)

    ## Creating the convolutional layer
    layer = tf.nn.conv2d(input=input,
                         filter=weights,
                         strides=[1, 1, 1, 1],
                         padding='SAME')

    layer += biases

    ## We shall be using max-pooling.
    if max_pooling is True:
        layer = tf.nn.max_pool(value=layer,
                               ksize=[1, 2, 2, 1],
                               strides=[1, 2, 2, 1],
                               padding='SAME')
    ## Output of pooling is fed to Relu which is the activation function for us.
    layer = tf.nn.relu(layer)

    return layer, weights


def create_flatten_layer(layer):
    # We know that the shape of the layer will be [batch_size img_size img_size num_channels]
    # But let's get it from the previous layer.
    layer_shape = layer.get_shape()

    ## Number of features will be img_height * img_width* num_channels. But we shall calculate it in place of hard-coding it.
    num_features = layer_shape[1:4].num_elements()

    ## Now, we Flatten the layer so we shall have to reshape to num_features
    layer = tf.reshape(layer, [-1, num_features])

    return layer


def create_fc_layer(input,
                    num_inputs,
                    num_outputs,
                    use_relu=True,
                    weight_loss=None):
    # Let's define trainable weights and biases.
    weights = create_weights(shape=[num_inputs, num_outputs], weight_loss=weight_loss)
    biases = create_biases(num_outputs)

    # Fully connected layer takes input x and produces wx+b.Since, these are matrices, we use matmul function in Tensorflow
    layer = tf.matmul(input, weights) + biases
    if use_relu:
        layer = tf.nn.relu(layer)

    return layer


layer_conv1, weights_conv1 = create_convolutional_layer(input=x,
                                         num_input_channels=num_channels,
                                         conv_filter_size=filter_size_conv1,
                                         num_filters=num_filters_conv1)

layer_conv2, weights_conv2 = create_convolutional_layer(input=layer_conv1,
                                         num_input_channels=num_filters_conv1,
                                         conv_filter_size=filter_size_conv2,
                                         num_filters=num_filters_conv2,
                                         max_pooling=False)

layer_conv3, weights_conv3 = create_convolutional_layer(input=layer_conv2,
                                         num_input_channels=num_filters_conv2,
                                         conv_filter_size=filter_size_conv3,
                                         num_filters=num_filters_conv3)

layer_conv4, weights_conv4 = create_convolutional_layer(input=layer_conv3,
                                         num_input_channels=num_filters_conv3,
                                         conv_filter_size=filter_size_conv4,
                                         num_filters=num_filters_conv4)

layer_flat = create_flatten_layer(layer_conv4)

layer_fc1 = create_fc_layer(input=layer_flat,
                            num_inputs=layer_flat.get_shape()[1:4].num_elements(),
                            num_outputs=fc_layer_size1,
                            use_relu=True,
                            weight_loss=0.04)
print('layer_fc1', layer_fc1)
layer_fc2 = create_fc_layer(input=layer_fc1,
                            num_inputs=fc_layer_size1,
                            num_outputs=fc_layer_size2,
                            use_relu=True,
                            weight_loss=0.04)
print('layer_fc2', layer_fc2)
layer_fc3 = create_fc_layer(input=layer_fc2,
                            num_inputs=fc_layer_size2,
                            num_outputs=num_classes,
                            use_relu=False)
print('layer_fc3', layer_fc3)
y_pred = tf.nn.softmax(layer_fc3, name='y_pred')

y_pred_cls = tf.argmax(y_pred, dimension=1)
session.run(tf.global_variables_initializer())
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=layer_fc3, labels=y_true)
cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')   # L2 Regularization
tf.add_to_collection('losses', cross_entropy_mean)
cost = tf.add_n(tf.get_collection('losses'), name='total_loss')
optimizer = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(cost)   # 1e-4
correct_prediction = tf.equal(y_pred_cls, y_true_cls)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
session.run(tf.global_variables_initializer())

print('y_pred', y_pred)
#print(session.run(y_pred))

def show_progress(epoch, feed_dict_train, feed_dict_validate, val_loss, duration=None):
    acc = session.run(accuracy, feed_dict=feed_dict_train)
    val_acc = session.run(accuracy, feed_dict=feed_dict_validate)
    if duration is not None:
        examples_per_sec = batch_size / duration
        msg = "Training Epoch {0}, Iterations: {1} --- Training Accuracy: {2:>6.1%}," \
              "  Validation Accuracy: {3:>6.1%},  Validation Loss: {4:.3f}," \
              " {5:.2f} examples/sec, {6:.2f} sec/iteration"
        #print(msg.format(epoch + 1, present_iterations, acc, val_acc, val_loss, examples_per_sec, duration))
    else:
        msg = "Training Epoch {0}, Iterations: {1} --- Training Accuracy: {2:>6.1%}," \
              "  Validation Accuracy: {3:>6.1%},  Validation Loss: {4:.3f}"
        #print(msg.format(epoch + 1, present_iterations, acc, val_acc, val_loss))
saver = tf.train.Saver()
saver.restore(sess=session, save_path='./save_model/fcs-model')
#saver.restore(sess=session, save_path='./fcs-model')


def plot_conv_weights(weights, input_channel=0):
    # weights_conv1 or weights_conv2.

    # 运行weights以获得权重
    w = session.run(weights)

    # 获取权重最小值最大值，这将用户纠正整个图像的颜色密集度，来进行对比
    w_min = np.min(w)
    w_max = np.max(w)

    # number of kernel
    print(w.shape)
    num_filters = w.shape[3]
    # 需要输出的卷积核
    #num_grids = math.ceil(math.sqrt(num_filters))
    num_grids = math.ceil(math.sqrt(num_filters))
    fig, axes = plt.subplots(num_grids, num_grids)
    fig.subplots_adjust(right=0.8)
    for i, ax in enumerate(axes.flat):
        # 只输出有用的子图.
        if i < num_filters:
            # 获得第i个卷积核在特定输入通道上的权重
            img = w[:, :, input_channel, i]

            cax = ax.imshow(img, vmin=w_min, vmax=w_max,
                      interpolation='nearest', cmap='coolwarm')   # seismic Spectral coolwarm inferno

        # 移除坐标.
        ax.set_xticks([])
        ax.set_yticks([])
    cax2 = plt.axes([0.85, 0.1, 0.02, 0.8])
    plt.colorbar(cax, cax=cax2, ax=ax)
    plt.show()


def plot_conv_layer(layer, image):
    # layer_conv1 or layer_conv2.

    # feed_dict只需要x，标签信息在此不需要.
    feed_dict = {x: [image]}

    # 获取该层的输出结果
    values = session.run(layer, feed_dict=feed_dict)

    # 卷积核树木
    num_filters = values.shape[3]

    # 每行需要输出的卷积核网格数
    #num_grids = math.ceil(math.sqrt(num_filters))
    if num_filters == 32:
        fig, axes = plt.subplots(4, 8)
    else:
        fig, axes = plt.subplots(8, 8)
    for i, ax in enumerate(axes.flat):
        # 只输出有用的子图.
        if i < num_filters:
            # 获取第i个卷积核的输出
            img = values[0, :, :, i]

            ax.imshow(img, interpolation='nearest', cmap='binary')   # binary Spectral inferno

        # 移除坐标.
        ax.set_xticks([])
        ax.set_yticks([])
    plt.show()


def plot_image(image):
    plt.imshow(image.reshape(32, 32, 3), interpolation='nearest', cmap='binary')
    plt.show()

filename = '../testing_data/fcs/5353.jpg'
filename = '../testing_data/track/4002.jpg'
image = cv2.imread(filename, 0)
image1 = cv2.resize(image, (40, 40), 0, 0, cv2.INTER_LINEAR)
image1 = np.array(image1).reshape(40, 40, 1)
#plot_conv_weights(weights=weights_conv1)
#plot_image(image)
#plot_conv_weights(weights=weights_conv2)
#plot_conv_weights(weights=weights_conv1, input_channel=1)
#plot_conv_weights(weights=weights_conv1, input_channel=2)
#plot_conv_weights(weights=weights_conv3, input_channel=0)
#plot_conv_weights(weights=weights_conv3, input_channel=1)
plot_conv_layer(layer=layer_conv1, image=image1)
plot_conv_layer(layer=layer_conv2, image=image1)
plot_conv_layer(layer=layer_conv3, image=image1)
plot_conv_layer(layer=layer_conv4, image=image1)
