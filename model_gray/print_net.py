import tensorflow as tf

# Prepare input data
classes = ['fcs', 'ground', 'track']
num_classes = len(classes)

# 20% of the data will automatically be used for validation
validation_size = 0.2
img_size = 400  #
num_channels = 1   # first conv layer input channel
train_path = '../training_data'

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


def create_weights2(shape, trainable=False):
    weights = tf.Variable(tf.Ones(shape))
    return weights


def create_biases(size):
    return tf.Variable(tf.constant(0.05, shape=[size]))


def create_convolutional_layer(input,
                               num_input_channels,
                               conv_filter_size,
                               num_filters,
                               max_pooling=True,
                               last_layer=False):
    if last_layer is True:
        weights = create_weights2(shape=[conv_filter_size, conv_filter_size, num_input_channels, num_filters])
        layer = tf.nn.conv2d(input=input,
                             filter=weights,
                             strides=[1, 1, 1, 1],
                             padding='SAME')
    else:
        weights = create_weights(shape=[conv_filter_size, conv_filter_size, num_input_channels, num_filters])
        biases = create_biases(num_filters)
        layer = tf.nn.conv2d(input=input,
                             filter=weights,
                             strides=[1, 1, 1, 1],
                             padding='SAME')
        layer += biases
        if max_pooling is True:
            layer = tf.nn.max_pool(value=layer,
                                   ksize=[1, 2, 2, 1],
                                   strides=[1, 2, 2, 1],
                                   padding='SAME')
        layer = tf.nn.relu(layer)
    return layer, weights


def create_flatten_layer(layer):
    layer_shape = layer.get_shape()
    num_features = layer_shape[1:4].num_elements()
    layer = tf.reshape(layer, [-1, num_features])
    return layer


def create_fc_layer(input,
                    num_inputs,
                    num_outputs,
                    use_relu=True,
                    weight_loss=None):
    weights = create_weights(shape=[num_inputs, num_outputs], weight_loss=weight_loss)
    biases = create_biases(num_outputs)
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

layer_conv_final = create_convolutional_layer(input=layer_conv4,
                                              num_input_channels=num_filters_conv4,
                                              conv_filter_size=5,
                                              num_filters=num_filters_conv4,
                                              last_layer=True)

layer_flat = create_flatten_layer(layer_conv_final)

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
                            num_outputs=50*50*num_classes,
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



session = tf.Session()
saver = tf.train.Saver()
saver.restore(sess=session, save_path='./save_msd/fcs-model')
w = session.run(weights_conv1)

def print_activations(t_ensor):
    print(t_ensor.op.name, '', t_ensor.get_shape().as_list())

print_activations(w)
