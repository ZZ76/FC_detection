import tensorflow as tf
import numpy as np
import os
import  glob
import cv2
import time
import sys, argparse
import matplotlib.pyplot as plt
import re

time_start_1 = time.clock()
src_img, src_img_grey, height, width = None, None, None, None
num, src_img_window, centers_map, centers_src_img = 2, None, None, None
num = 0

step_size = (5, 5, 5)
window_size = (100, 80, 60)
#window_size = (50, 40, 30)
#step_size = (5, 10)
#window_size = (100, 90)
threshold = 0.90   #0.98
center_threshold = 0.90
images = []
locations = []
image_size = 40
num_channels = 1
result = None
n_percenter = None
center_range = range(80, 200, 12)
#k_gau = [(3, 3), (5, 5), (7, 7), (9, 9), (11, 11)]
k_gau = [(3, 3), (7, 7), (11, 11)]   # gausian kernel for select() to generate location
#center_range = range(30, 50, 3)


def select():
    c = 0
    img2 = src_img.copy()
    img2 = cv2.fastNlMeansDenoisingColored(img2, None, 4, 4, 5, 9)
    img3 = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)
    for k in k_gau:
        img2 = cv2.GaussianBlur(img3, (5, 5), 0)
        img2 = cv2.Laplacian(img2, cv2.CV_64F)
        img2 = 255 - img2
        img2 = cv2.GaussianBlur(img2, k, 0)
        img2 = np.uint8(img2)
        dist_transform = cv2.distanceTransform(img2, cv2.DIST_L2, 5)
        _, thresh = cv2.threshold(dist_transform, 12, 255, cv2.THRESH_BINARY)
        thresh = np.uint8(thresh)
        _, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        for i in contours:
            x, y, w, h = cv2.boundingRect(i)
            if w < 20 or h < 20 or w > 200 or h > 200:
                continue
            cv2.circle(centers_map, (int((2 * x + w) / 2), int((2 * y + h) / 2)), 8, (0, 0, 0), -1, lineType=4)
            cv2.circle(centers_src_img, (int((2 * x + w) / 2), int((2 * y + h) / 2)), 8, (0, 0, 0), -1, lineType=4)
            c = c + 1
    #print('center_num =', c)


def select2():
    c = 0
    img2 = src_img.copy()
    img2 = cv2.fastNlMeansDenoisingColored(img2, None, 4, 4, 5, 9)
    img3 = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)
    for k in k_gau:
        img2 = cv2.GaussianBlur(img3, (5, 5), 0)
        img2 = cv2.Laplacian(img2, cv2.CV_64F)
        img2 = 255 - img2
        img2 = cv2.GaussianBlur(img2, k, 0)
        img2 = np.uint8(img2)
        dist_transform = cv2.distanceTransform(img2, cv2.DIST_L2, 5)
        _, thresh = cv2.threshold(dist_transform, 8, 255, cv2.THRESH_BINARY)
        kernel = np.ones((3, 3), np.uint8)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=5)
        thresh = np.uint8(thresh)
        _, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        for i in contours:
            x, y, w, h = cv2.boundingRect(i)
            if w < 20 or h < 20 or w > 180 or h > 180:
                continue
            cv2.circle(centers_map, (int((2 * x + w) / 2), int((2 * y + h) / 2)), 8, (0, 0, 0), -1, lineType=4)
            cv2.circle(centers_src_img, (int((2 * x + w) / 2), int((2 * y + h) / 2)), 8, (0, 0, 0), -1, lineType=4)
            c = c + 1


def predict(imagelist):
    print('Length is', len(imagelist))
    start = time.clock()
    imagelist = np.array(imagelist, dtype=np.uint8)
    imagelist = imagelist.astype('float32')
    imagelist = np.multiply(imagelist, 1.0/255.0)
    #The input to the network is of shape [None image_size image_size num_channels]. Hence we reshape.
    x_batch = imagelist.reshape(len(locations), image_size, image_size, num_channels)   #first variable is number of images
    ## Let us restore the saved model
    tf.reset_default_graph()   # finalize() after tf.session()
    with tf.Session() as sess:
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
    tf.get_default_graph().finalize()   # important!!! loop will be slower without this line
    # result is of this format [probabiliy_of_rose probability_of_sunflower]
    end = time.clock()
    print('predict =', end - start)


def putsquares(x1, x2, y1, y2):
    blank_image[y1:y2, x1:x2] = blank_image[y1:y2, x1:x2] + 1


def putlocations():
    start = time.clock()
    for i in range(0, len(locations)):
        x1, y1, x2, y2 = locations[i][0], locations[i][1], locations[i][2], locations[i][3]
        if result[i][0] >= threshold:   # for different classes
            #global boxes_img
            #cv2.rectangle(boxes_img, (x1, y1), (x2, y2), (0, 255, 0), 1)
            cv2.circle(centers_map, (int((x1 + x2)/2), int((y1 + y2)/2)), 4, (0, 0, 0), -1, lineType=4)
            #putsquares(x1, x2, y1, y2)
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
    global center_list
    global images
    images = []
    global locations
    locations = []
    #print('images', images)
    for i in center_cnt:
        area = cv2.contourArea(i)
        x, y, w, h = cv2.boundingRect(i)
        #cv2.putText(centers_map, str(area), (x, int(y * 0.95)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
        if area <= 5000:
            moments = cv2.moments(i)
            if moments['m00'] == 0:
                continue
            else:
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
    global n_percenter
    n_percenter = 0   # how many pics or results for each center
    for c in center_range:   #center_range
        n_percenter = n_percenter + 1
    print('n_percenter', n_percenter)
    for i in range(0, len(center_list)):   # loop in each center   range  from 1???
        plist = []   # a group of value from result[]
        for j in range(0, n_percenter):   # find max in result[]
            l = i * n_percenter + j   # index in centerlist and result i - 1???
            plist.append(result[l][0])
        if max(plist) >= center_threshold:
            print('max index', plist.index(max(plist)))
            print('max value', max(plist))
            l = i * n_percenter + plist.index(max(plist))
            x1, y1, x2, y2 = locations[l][0], locations[l][1], locations[l][2], locations[l][3]
            if x1 < 0:
                x1 = 0
            if y1 < 0:
                y1 = 0
            if x2 >= width:
                x2 = width - 1
            if y2 >= height:
                y2 = height - 1
            if save is True:
                window = src_img_window[y1:y2, x1:x2]
                window = cv2.resize(window, (100, 100))
                name = str(num) + '_' + str(i) + '.jpg'
                cv2.imwrite(save_path + '/' + name, window)
            #cv2.rectangle(src_img, (x1, y1), (x2, y2), (0, 255, 0), 1)
            #cv2.putText(src_img, str(i), (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
            cv2.rectangle(centers_src_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(centers_src_img, str(i), (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
            print(i, 'result', plist)
            print()
        #print(i, 'result', plist)
        #print()

def filterboxes(save=False):
    start = time.clock()
    global images, locations, result
    countcontours()   # find center of contour, build center_list[], pick images
    #print('centerlist', center_list)
    print('Potential_FCs_num =', len(center_list))
    predict(images)   # return result[]
    if save is True:
        putfinallocations(save=True)   # this step save predicted windows
    else:
        putfinallocations()
    images, locations, result = [], [], None
    end = time.clock()
    print('filter time =', end - start)

def plot_image(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    plt.imshow(image)
    plt.show()


def locate2(imgfullpath, show_result=False, save=False, save_small=False):
    global locations, result, images, src_img, src_img_grey, height, width, src_img_window
    global blank_image, centers_map, center_list, centers_src_img
    src_img = cv2.imread(imgfullpath)
    centers_src_img = src_img.copy()
    #boxes_img = src_img.copy()
    src_img_window = src_img.copy()
    src_img_grey = cv2.cvtColor(src_img, cv2.COLOR_RGB2GRAY)
    height, width = src_img.shape[:2]
    #print(width, height)
    #blank_image = np.zeros((height, width, 1), np.uint8)  # for showing the map
    centers_map = np.zeros((height, width, 1), np.uint8)
    centers_map[:] = 255
    center_list = np.array([], np.int32, ndmin=2)
    center_list = np.reshape(center_list, (-1, 2))
    select()
    if save_small is True:
        filterboxes(save=True)
    else:
        filterboxes()
    if save is True:
        name = str(num) + '_' + 'result' + '.bmp'
        cv2.imwrite(save_path + '/' + name, centers_src_img)
    if show_result is True:
        cv2.imshow('center', centers_src_img)
        cv2.imshow(imgfullpath, src_img)


def locate2_2(imgfullpath, show_result=False, save=False, save_small=False):
    global locations, result, images, src_img, src_img_grey, height, width, src_img_window
    global blank_image, centers_map, center_list, centers_src_img
    src_img = cv2.imread(imgfullpath)
    centers_src_img = src_img.copy()
    #boxes_img = src_img.copy()
    src_img_window = src_img.copy()
    src_img_grey = cv2.cvtColor(src_img, cv2.COLOR_RGB2GRAY)
    height, width = src_img.shape[:2]
    #print(width, height)
    #blank_image = np.zeros((height, width, 1), np.uint8)  # for showing the map
    centers_map = np.zeros((height, width, 1), np.uint8)
    centers_map[:] = 255
    center_list = np.array([], np.int32, ndmin=2)
    center_list = np.reshape(center_list, (-1, 2))
    select2()
    if save_small is True:
        filterboxes(save=True)
    else:
        filterboxes()
    if save is True:
        name = str(num) + '_' + 'result' + '.bmp'
        cv2.imwrite(save_path + '/' + name, centers_src_img)
    if show_result is True:
        cv2.imshow('center', centers_src_img)
        cv2.imshow(imgfullpath, src_img)


def read_filse_and_locate():
    img_files = os.listdir(img_folder_path)
    global num
    for f in img_files:
        if f[-4:] == '.TIF' or f[-4:] == '.jpg':
            time_start_2 = time.clock()
            num = re.findall('\d+', f)[-1]
            img_full_path = img_folder_path + '/' + f
            print('\n''NEW')
            print(img_full_path)
            locate2(img_full_path, show_result=False, save=True)
            #locate2_2(img_full_path, show_result=False, save=True)
            time_end_2 = time.clock()
            print('one img time =', time_end_2 - time_start_2)


image_name = '../single_testing_images/DFC/DFC_AOI_RGB_tile335.TIF'   # DFC
image_name = '../single_testing_images/DFC3/Escourt_dfc_29April2017_rgb_transparent_mosaic_group1327.TIF'
#image_name = '../single_testing_images/GFC/GFC_AOI6_RGB_tile631.TIF'   # GFC
#image_name = '../single_testing_images/compare2/DSC09886.jpg'
#image_name = '../single_testing_images/ng/ng5.jpg'
img_folder_path = '../single_testing_images/DFC'
#img_folder_path = '../single_testing_images/compare2'
#img_folder_path = '../single_testing_images/ng'
save_path = './save2'
#save_path = './save2'

#read_filse_and_locate()   # save and do not show by default
locate2(image_name, show_result=True, save=False)
#locate2_2(image_name, show_result=True, save=False)
time_end = time.clock()
print('total time =', time_end - time_start_1)
cv2.waitKey(0)
cv2.destroyAllWindows()
