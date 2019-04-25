import os
import math
import random

import numpy as np
import tensorflow as tf
import cv2
import time
import matplotlib.image as mpimg

slim = tf.contrib.slim
import sys

sys.path.append('./')
from nets import ssd_vgg_300, ssd_common, np_methods
from nets import nets_factory
from preprocessing import ssd_vgg_preprocessing
from notebooks import visualization


def process_image(img, select_threshold=0.5, nms_threshold=0.5, net_shape=(300, 300)):
    rimg, rpredictions, rlocalisations, rbbox_img = sess.run(
        [image_4d, predictions, localisations, bbox_img], feed_dict={img_input: img})
    rclasses, rscores, rbboxes = np_methods.ssd_bboxes_select(
        rpredictions, rlocalisations, ssd_anchors,
        select_threshold=select_threshold, img_shape=net_shape, num_classes=3, decode=True)
    rbboxes = np_methods.bboxes_clip(rbbox_img, rbboxes)
    rclasses, rscores, rbboxes = np_methods.bboxes_sort(rclasses, rscores, rbboxes, top_k=400)
    rclasses, rscores, rbboxes = np_methods.bboxes_nms(rclasses, rscores, rbboxes, nms_threshold=nms_threshold)
    rbboxes = np_methods.bboxes_resize(rbbox_img, rbboxes)
    return rclasses, rscores, rbboxes

gpu_options = tf.GPUOptions(allow_growth=False)
config = tf.ConfigProto(log_device_placement=False, gpu_options=gpu_options)

#################################
with tf.Session() as sess:
    net_shape = (300, 300)
    data_format = 'NHWC'
    img_input = tf.placeholder(tf.uint8, shape=(None, None, 3))
    image_pre, labels_pre, bboxes_pre, bbox_img = ssd_vgg_preprocessing.preprocess_for_eval(
        img_input, None, None, net_shape, data_format, resize=ssd_vgg_preprocessing.Resize.WARP_RESIZE)
    image_4d = tf.expand_dims(image_pre, 0)

    reuse = True if 'ssd_net' in locals() else None

    ssd_class = nets_factory.get_network('ssd_300_vgg')
    ssd_params = ssd_class.default_params._replace(num_classes=3)
    # ssd_net = ssd_vgg_300.SSDNet()
    ssd_net = ssd_class(ssd_params)
    with slim.arg_scope(ssd_net.arg_scope(data_format=data_format)):
        predictions, localisations, _, _ = ssd_net.net(image_4d, is_training=False, reuse=reuse)
    ckpt_filename = './logs_unt_aerial_dataset_20190103_4_0.3/model.ckpt-21274'

    sess.run(tf.global_variables_initializer())

    saver = tf.train.Saver()
    saver.restore(sess, ckpt_filename)
    ssd_anchors = ssd_net.anchors(net_shape)

    # image_path = '../demo_datas/images/action_3_2.png'
    # image_names = sorted(os.listdir())
    # img = mpimg.imread(image_path)
    # rclasses, rscores, rbboxes = process_image(img)
    # visualization.plt_bboxes(img, rclasses, rscores, rbboxes)


    video_path = '../demo_datas/actions3.mpg'
    capture = cv2.VideoCapture(video_path)
    ii = 0
    frame_rate_divider = 2  # for skiping frame formula= 60/fps
    while (capture.isOpened()):
        stime = time.time()  # for timing the frame time
        ret, frame = capture.read()  # ret is true or false (if video is playing then its true)
        cv2.imwrite('/home/yq0033/work/PycharmProjects/SSD/demo_datas/labeled/actions2_%d.png' % ii, frame)
        if ii % frame_rate_divider == 0:
            rclasses, rscores, rbboxes = process_image(frame)
            if ret:
                img = visualization.plt_bboxes1(frame, rclasses, rscores, rbboxes)
                ii += 1
                cv2.imwrite('/home/yq0033/work/PycharmProjects/SSD/demo_datas/labeled/actions2_labeled_%d.png' % ii, img)
        else:
            ii += 1

        if cv2.waitKey(1) & 0xFF == ord('q'):  # if we hit the "Q" key it will go to next line
            break

    capture.release()
    cv2.destroyAllWindows()