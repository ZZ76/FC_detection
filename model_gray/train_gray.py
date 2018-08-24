import dataset_gray
import tensorflow as tf
import time
from datetime import timedelta
import math
import random
import numpy as np

# Adding Seed so that random initialization is consistent
from numpy.random import seed

seed(1)
from tensorflow import set_random_seed

set_random_seed(2)

batch_size = 2048   # 32 3000 4000

# Prepare input data
classes = ['fcs', 'ground', 'track']
num_classes = len(classes)

# 20% of the data will automatically be used for validation
validation_size = 0.3
img_size = 40  #
num_channels = 1   # first conv layer input channel
train_path = '../training_data'

# We shall load all the training and validation images and labels into memory using openCV and use that during training
data = dataset_gray.read_train_sets(train_path, img_size, classes, validation_size=validation_size)

print("Complete reading input data. Will Now print a snippet of it")
print("Number of files in Training-set:\t\t{}".format(len(data.train.labels)))
print("Number of files in Validation-set:\t{}".format(len(data.valid.labels)))

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

    return layer


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


layer_conv1 = create_convolutional_layer(input=x,
                                         num_input_channels=num_channels,
                                         conv_filter_size=filter_size_conv1,
                                         num_filters=num_filters_conv1)

layer_conv2 = create_convolutional_layer(input=layer_conv1,
                                         num_input_channels=num_filters_conv1,
                                         conv_filter_size=filter_size_conv2,
                                         num_filters=num_filters_conv2,
                                         max_pooling=False)

layer_conv3 = create_convolutional_layer(input=layer_conv2,
                                         num_input_channels=num_filters_conv2,
                                         conv_filter_size=filter_size_conv3,
                                         num_filters=num_filters_conv3)

layer_conv4 = create_convolutional_layer(input=layer_conv3,
                                         num_input_channels=num_filters_conv3,
                                         conv_filter_size=filter_size_conv4,
                                         num_filters=num_filters_conv4)

layer_flat = create_flatten_layer(layer_conv4)

layer_fc1 = create_fc_layer(input=layer_flat,
                            num_inputs=layer_flat.get_shape()[1:4].num_elements(),
                            num_outputs=fc_layer_size1,
                            use_relu=True,
                            weight_loss=0.04)

layer_fc2 = create_fc_layer(input=layer_fc1,
                            num_inputs=fc_layer_size1,
                            num_outputs=fc_layer_size2,
                            use_relu=True,
                            weight_loss=0.04)

layer_fc3 = create_fc_layer(input=layer_fc2,
                            num_inputs=fc_layer_size2,
                            num_outputs=num_classes,
                            use_relu=False)

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


def show_progress(epoch, feed_dict_train, feed_dict_validate, val_loss, duration=None):
    acc = session.run(accuracy, feed_dict=feed_dict_train)
    val_acc = session.run(accuracy, feed_dict=feed_dict_validate)
    if duration is not None:
        examples_per_sec = batch_size / duration
        msg = "Training Epoch {0}, Iterations: {1} --- Training Accuracy: {2:>6.1%}," \
              "  Validation Accuracy: {3:>6.1%},  Validation Loss: {4:.3f}," \
              " {5:.2f} examples/sec, {6:.2f} sec/iteration"
        print(msg.format(epoch + 1, present_iterations, acc, val_acc, val_loss, examples_per_sec, duration))
    else:
        msg = "Training Epoch {0}, Iterations: {1} --- Training Accuracy: {2:>6.1%}," \
              "  Validation Accuracy: {3:>6.1%},  Validation Loss: {4:.3f}"
        print(msg.format(epoch + 1, present_iterations, acc, val_acc, val_loss))


total_iterations = 0
present_iterations = 0
saver = tf.train.Saver()
loss_min = 1

def train(num_iteration, continue_training=True):
    global total_iterations
    global present_iterations
    global saver
    global loss_min
    if continue_training is True:
        #saver = tf.train.import_meta_graph('./model1/fcs-model.meta')
        saver.restore(session, tf.train.latest_checkpoint('./save_model/'))
    for i in range(total_iterations, total_iterations + num_iteration):
        start_time = time.clock()
        present_iterations = i
        x_batch, y_true_batch, _, cls_batch = data.train.next_batch(batch_size)
        x_batch = np.array(x_batch).reshape(batch_size, img_size, img_size, num_channels)   # reshape
        # for only 1 channel reshape to [16,64,64,1], or it will be [16,64,64]
        x_valid_batch, y_valid_batch, _, valid_cls_batch = data.valid.next_batch(batch_size)
        x_valid_batch = np.array(x_valid_batch).reshape(batch_size, img_size, img_size, num_channels)   # reshape
        feed_dict_tr = {x: x_batch,
                        y_true: y_true_batch}
        feed_dict_val = {x: x_valid_batch,
                         y_true: y_valid_batch}

        session.run(optimizer, feed_dict=feed_dict_tr)
        val_loss = session.run(cost, feed_dict=feed_dict_val)
        epoch = int(i / int(data.train.num_examples / batch_size))
        if i == 0:
            loss_min = val_loss
            iterations_duration = time.clock() - start_time
        if val_loss < loss_min:
            loss_min = val_loss
            saver.save(session, './save_model/fcs-model')
            show_progress(epoch, feed_dict_tr, feed_dict_val, val_loss, duration=iterations_duration)
        if i % int(data.train.num_examples / batch_size) == 0:
            #val_loss = session.run(cost, feed_dict=feed_dict_val)
            #epoch = int(i / int(data.train.num_examples / batch_size))
            show_progress(epoch, feed_dict_tr, feed_dict_val, val_loss, duration=iterations_duration)
            #saver.save(session, './model1/fcs-model')

    total_iterations += num_iteration

#train(num_iteration=2000, continue_training=False)   # new training
train(num_iteration=300)   #3000
