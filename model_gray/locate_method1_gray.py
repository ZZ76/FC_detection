import tensorflow as tf
import numpy as np
import os, glob, cv2
import time
import sys, argparse
import matplotlib.pyplot as plt
import re
from matplotlib.pylab import cm

time_start = time.clock()
src_img, src_img_grey, height, width = None, None, None, None
num, src_img_window, blank_image, blank_image2, centers_map = 'p1', None, None, None, None
step_size = (5, 5, 5)
window_size = (100, 80, 60)
#window_size = (50, 40, 30)
#step_size = (5, 10)
#window_size = (100, 90)
threshold = 0.85   #0.98
center_threshold = 0.95
images = []
locations = []
image_size = 40   #128
num_channels = 1
result = None
center_area = 140   # 190 260 280 650
center_list = None
n_percenter = None
center_range = range(50, 100, 4)
#center_range = range(30, 50, 3)


def showslide(i, m):
    cv2.imshow('mask', m)
    #mask[:] = i[:]


def slide(ws, ss):   # ws = window_size, ss = step_size
    flag_y = True
    x1 = 0   # 0
    y1 = 0   # 0
    x2 = ws   # ws
    y2 = ws   # ws
    start = time.clock()
    while flag_y is True:
        flag_x = True
        while flag_x is True:
            region = src_img_grey[y1:y2, x1:x2]
            locations.append((x1, y1, x2, y2))
            #region = cv2.resize(region, (image_size, image_size), 0, 0, cv2.INTER_LINEAR)
            region = cv2.resize(region, (image_size, image_size))
            global images
            images.append(region)
            #m = cv2.rectangle(m, (x1, y1), (x2, y2), (0, 255, 0), 2)   # drawing green sliding window
            x1 = x1 + ss
            x2 = x2 + ss
            if x2 > width:   # x2 for right border
                x1, x2 = 0, ws
                y1 = y1 + ss
                y2 = y2 + ss
                flag_x = False
            if y2 > height:   # y2 for bottom border
                flag_x = False
                flag_y = False
    end = time.clock()
    print('slide =', end - start)


def predict(imagelist):
    imagelist = np.array(imagelist, dtype=np.uint8)
    imagelist = imagelist.astype('float32')
    imagelist = np.multiply(imagelist, 1.0 / 255.0)
    x_batch = imagelist.reshape(len(locations), image_size, image_size,
                                num_channels)
    tf.reset_default_graph()  # finalize() after tf.session()
    sess = tf.Session()
    saver = tf.train.import_meta_graph('./save_model/fcs-model.meta')
    saver.restore(sess, tf.train.latest_checkpoint('./save_model/'))
    graph = tf.get_default_graph()
    y_pred = graph.get_tensor_by_name("y_pred:0")
    x = graph.get_tensor_by_name("x:0")
    y_true = graph.get_tensor_by_name("y_true:0")
    y_test_images = np.zeros((1, 3))
    ### Creating the feed_dict that is required to be fed to calculate y_pred
    feed_dict_testing = {x: x_batch, y_true: y_test_images}
    global result
    result = sess.run(y_pred, feed_dict=feed_dict_testing)
    # result is of this format [probabiliy_of_rose probability_of_sunflower]
    sess.close()
    tf.get_default_graph().finalize()  # important!!! loop will be slower without this line

def predict(imagelist):
    print('length is', len(imagelist))
    start = time.clock()
    if len(imagelist) >= 4000:
        imagelist1 = imagelist[:int(len(imagelist)/2)]
        imagelist = imagelist[int(len(imagelist)/2):]
    imagelist = np.array(imagelist, dtype=np.uint8)
    imagelist = imagelist.astype('float32')
    imagelist = np.multiply(imagelist, 1.0/255.0)
    #The input to the network is of shape [None image_size image_size num_channels]. Hence we reshape.
    x_batch = imagelist.reshape(len(locations), image_size, image_size, num_channels)   #first variable is number of images
    ## Let us restore the saved model
    tf.reset_default_graph()  # finalize() after tf.session()
    sess = tf.Session()
    # Step-1: Recreate the network graph. At this step only graph is created.
    saver = tf.train.import_meta_graph('./save_model/fcs-model.meta')
    # Step-2: Now let's load the weights saved using the restore method.
    saver.restore(sess, tf.train.latest_checkpoint('./save_model/'))

    # Accessing the default graph which we have restored
    graph = tf.get_default_graph()

    # Now, let's get hold of the op that we can be processed to get the output.
    # In the original network y_pred is the tensor that is the prediction of the network
    y_pred = graph.get_tensor_by_name("y_pred:0")

    ## Let's feed the images to the input placeholders
    x = graph.get_tensor_by_name("x:0")
    y_true = graph.get_tensor_by_name("y_true:0")
    y_test_images = np.zeros((1, 3))
    ### Creating the feed_dict that is required to be fed to calculate y_pred
    feed_dict_testing = {x: x_batch, y_true: y_test_images}
    global result
    result = sess.run(y_pred, feed_dict=feed_dict_testing)
    # result is of this format [probabiliy_of_rose probability_of_sunflower]
    sess.close()
    tf.get_default_graph().finalize()  # important!!! loop will be slower without this line
    end = time.clock()
    print('predict =', end - start)


def putsquares(x1, x2, y1, y2):
    blank_image[y1:y2, x1:x2] = blank_image[y1:y2, x1:x2] + 1


def putlocations():
    start = time.clock()
    for i in range(0, len(locations)):
        x1, y1, x2, y2 = locations[i][0], locations[i][1], locations[i][2], locations[i][3]
        if result[i][0] >= threshold:   # for different classes
            global boxes_img
            cv2.rectangle(boxes_img, (x1, y1), (x2, y2), (0, 255, 0), 1)
            cv2.circle(centers_map, (int((x1 + x2)/2), int((y1 + y2)/2)), 4, (0, 0, 0), -1, lineType=4)
            putsquares(x1, x2, y1, y2)
    end = time.clock()
    print('putlocations =', end - start)


def pickfromcenter(cx, cy):   # pick a series of images put into images[]
    for i in center_range:   #center_range
        x1, x2 = int(cx - i/2), int(cx + i/2)
        y1, y2 = int(cy - i/2), int(cy + i/2)
        if x1 < 0:
            x1 = 0
        if y1 < 0:
            y1 = 0
        if x2 >= width:
            x2 = width - 1
        if y2 >= height:
            y2 = height - 1
        region = src_img_grey[y1:y2, x1:x2]
        locations.append((x1, y1, x2, y2))
        region = cv2.resize(region, (image_size, image_size))
        images.append(region)


def countcontours():   # find center of each contour put into center_list[], then pick images use center
    _, center_cnt, _ = cv2.findContours(centers_map, 1, 2)
    global center_list, images, locations
    images = []
    locations = []
    #print('images', images)
    for i in center_cnt:
        area = cv2.contourArea(i)
        x, y, w, h = cv2.boundingRect(i)
        cv2.putText(centers_map, str(area), (x, int(y * 0.95)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
        if center_area <= area <= 5000:
            moments = cv2.moments(i)
            cx = int(moments['m10'] / moments['m00'])
            cy = int(moments['m01'] / moments['m00'])
            center_list = np.append(center_list, [[cx, cy]], axis=0)
            #center_list = np.append(center_list, (cx, cy))
            #center_list.append((cx, cy))
            pickfromcenter(cx, cy)
            #cv2.rectangle(src_img, (cx - 30, cy - 30), (cx + 30, cy + 30), (0, 255, 0), 1)
        else:
            pass


def putfinallocations(save=False):
    global n_percenter, src_img
    n_percenter = 0   # how many pics or results for each center
    for c in center_range:   #center_range
        n_percenter = n_percenter + 1
    print('n_percenter', n_percenter)
    for i in range(0, len(center_list)):   # loop in each center   range  from 1???
        plist = []   # probability list, a group of value from result[]
        for j in range(0, n_percenter):   # find max in result[]
            l = i * n_percenter + j   # index in centerlist and result i - 1???
            plist.append(result[l][0])
        if max(plist) >= center_threshold:
            #print('l', l)
            print('max index', plist.index(max(plist)))
            print('max value', max(plist))
            l = i * n_percenter + plist.index(max(plist))
            x1, y1, x2, y2 = locations[l][0], locations[l][1], locations[l][2], locations[l][3]
            if save is True:
                window = src_img_window[2*y1:2*y2, 2*x1:2*x2]   # 2*(x, y), because the size was cut to half
                window = cv2.resize(window, (100, 100))
                name = str(num) + '_' + str(i) + '.jpg'
                cv2.imwrite(save_path + '/' + name, window)
            cv2.rectangle(src_img, (x1, y1), (x2, y2), (0, 255, 0), 1)
            cv2.putText(src_img, str(i), (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
        else:
            pass
        print(i, 'result', plist)
        print()

def filterboxes(save=False):
    start = time.clock()
    global images, locations, result
    countcontours()   # find center of contour, build center_list[], pick images
    print('centerlist', center_list)
    print('Potential_FCs_num =', len(center_list))
    predict(images)   # return result[]
    if save is True:
        putfinallocations(save=True)   # this step save predicted windows
    else:
        putfinallocations()
    end = time.clock()
    images = []
    locations = []
    result = None
    print('filter time =', end - start)

def plot_image(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    plt.imshow(image)
    plt.show()

def trans_colormap(src_img, colormap=cm.cubehelix):   # inferno plasma magma cubehelix
    if src_img.shape[2] == 3:
        src_img = cv2.cvtColor(src_img, cv2.COLOR_RGB2GRAY)
    colorized = np.reshape(src_img, (height, width))
    colorized = colormap(colorized)
    colorized = (colorized * 255).astype(np.uint8)
    colorized = cv2.cvtColor(colorized, cv2.COLOR_BGR2RGB)   # to show in opencv
    print(colorized.shape)
    return colorized

def locate1(imgfullpath, show_result=False, save=False, save_small=False):
    global locations, result, images, src_img, src_img_grey, height, width, src_img_window, boxes_img
    global blank_image, blank_image2, centers_map, center_list
    src_img = cv2.imread(imgfullpath)
    src_img_window = src_img.copy()
    height, width = src_img.shape[:2]
    height, width = int(height / 2), int(width / 2)
    src_img = cv2.resize(src_img, (width, height))
    src_img_grey = cv2.cvtColor(src_img, cv2.COLOR_RGB2GRAY)
    print(width, height)
    blank_image = np.zeros((height, width, 1), np.uint8)  # for showing the map
    centers_map = np.zeros((height, width, 1), np.uint8)
    centers_map[:] = 255
    center_list = np.array([], np.int32, ndmin=2)
    center_list = np.reshape(center_list, (-1, 2))
    boxes_img = src_img.copy()
    blank_image2 = blank_image.copy()
    for j in range(0, len(window_size)):
        print(window_size[j], step_size[j])
        slide(window_size[j], step_size[j])
        predict(images)
        putlocations()
        images = []
        locations = []
        result = None
    if save_small is True:
        filterboxes(save=True)
    else:
        filterboxes()
    ### for generating big image
    #blank_image = cv2.equalizeHist(blank_image)
    #print(blank_image2.shape)
    blank_image = trans_colormap(blank_image)
    #blank_image2 = cv2.applyColorMap(blank_image2, cv2.COLORMAP_RAINBOW)
    #overlapped = cv2.addWeighted(src_img, 0.8, blank_image, 0.6, 0)
    new_img = np.zeros((2 * height, 2 * width, 3), np.uint8)   # 3 channels
    new_img[0:height, 0:width] = src_img
    #cv2.imshow('boxes', boxes_img)
    #boxes_img = np.array(boxes_img).reshape(height, width, 1)   # reshape
    new_img[0:height, width:2 * width] = boxes_img
    new_img[height:2 * height, 0:width] = blank_image
    #new_img[height:2 * height, width:2 * width] = overlapped
    new_img[height:2 * height, width:2 * width] = centers_map
    if save is True:
        name = str(num) + '_' + 'result' + '.bmp'
        cv2.imwrite(save_path + '/' + name, new_img)
    if show_result is True:
        cv2.imshow(image_name, new_img)
    time_end = time.clock()
    print('total time =', time_end - time_start)


def read_filse_and_locate():
    img_files = os.listdir(img_folder_path)
    global num
    for f in img_files:
        if f[-4:] == '.jpg':
            num = re.findall('\d+', f)[-1]
            img_full_path = img_folder_path + '/' + f
            print('\n''NEW')
            print(img_full_path)
            locate1(img_full_path, show_result=False, save=True, save_small=True)


image_name = '../single_testing_images/DFC/DFC_AOI_RGB_tile367.TIF'   # DFC
#image_name = '../single_testing_images/DFC2/DFC_AOI_RGB_tile305.TIF'
#image_name = '../single_testing_images/GFC/GFC_AOI6_RGB_tile658.TIF'   # GFC
image_name = '../single_testing_images/ng/ng1.jpg'
img_folder_path = '../single_testing_images/compare2'
#img_folder_path = '../single_testing_images/DFC'
#save_path = './save1/d2'
save_path = './save1'

#read_filse_and_locate()   # save and do not show by default
locate1(image_name, show_result=True, save=False)

cv2.waitKey(0)
cv2.destroyAllWindows()
