"""
2019-03-08
Read parameter values from checkpoints.
"""
import os
import tensorflow as tf
import numpy as np
from tensorflow.python import pywrap_tensorflow
from tensorflow.python.tools.inspect_checkpoint import print_tensors_in_checkpoint_file

SSD_300_VGG = './checkpoints/'
# CHECKPOINT_PATH = './logs_20190307/RESIZED_PASCAL_VOC_2007_20190307_Trainable_Losses'
# CHECKPOINT_PATH = './logs_20190301/PASCAL_VOC_2007_20190301_Trainable_Losses'
# CHECKPOINT_PATH = './logs_20190307/RESIZED_PASCAL_VOC_2007_20190307_Trainable_Losses/'
# CHECKPOINT_PATH = './logs_20190303/PASCAL_VOC_2012_20190304_Trainable_Losses_1/'
# CHECKPOINT_PATH = './logs_20190307/PASCAL_VOC_2007_20190307_Trainable_Losses_0.01-1/'
# CHECKPOINT_PATH = './logs_20190307/PASCAL_VOC_2007_20190307_Trainable_Losses_0.2-1/'
# CHECKPOINT_PATH = './logs_20190317/PASCAL_VOC_0712_20190317_Original/'

# CHECKPOINT_PATH = './logs/PASCAL_VOC_2007/Fixed_With_ckpt_0.01_0.1_0.3_0.6_0.5_1/model.ckpt-25000'
CHECKPOINT_PATH = './logs/PASCAL_VOC_2007/Original_With_ckpt/model.ckpt-25000'
# CHECKPOINT_PATH = './logs/PASCAL_VOC_2007/Fixed_With_ckpt_0.01_0.1_0.3_0.6_0.5_1/model.ckpt-25000'

# TRAIN_DIR = './logs/PASCAL_VOC_2007/Original_With_ckpt_SGD_Triangular'
# TRAIN_DIR = './logs/PASCAL_VOC_2007/20190413_Original_With_ckpt_SGD_Triangular_summaries=6'

TRAIN_DIR = './logs/UNT_UAV_Dataset/20190412_With_ckpt_Fixed_Person_Car_0.3/'


def print_values(dir, steps):
    checkpoint_file = os.path.join(dir, 'model.ckpt-' + str(steps))
    print_tensors_in_checkpoint_file(file_name=checkpoint_file, tensor_name='', all_tensors=True)
    print(checkpoint_file)


def print_values1(dir, _filename):
    checkpoint_file = os.path.join(dir, _filename)
    print_tensors_in_checkpoint_file(file_name=checkpoint_file,
                                     tensor_name='ssd_300_vgg/conv1/conv1_1/biases',
                                     all_tensors=False)


def inspect_ckpt1(ckpt_path):
    tf.reset_default_graph()
    saver = tf.train.import_meta_graph(ckpt_path)
    grapd_def = tf.get_default_graph().as_graph_def()
    node_list = [n.name for n in grapd_def.node]
    print("################################")
    for node in node_list:
        if node.startswith("ssd_300_vgg/conv1/conv1_1/"):
            print("##: ", type(node), node)
    print("################################")


def inspect_ckpt(checkpoint_path):
    reader = pywrap_tensorflow.NewCheckpointReader(checkpoint_path)
    var_to_shape_map = reader.get_variable_to_shape_map()
    # print(var_to_shape_map)
    # keys = var_to_shape_map.keys()
    # sorted_keys = sorted(keys)
    # for key in sorted_keys:
    #     print(key)
    keys = [key for key in var_to_shape_map if key.startswith("ssd_300_vgg/conv7/")]
    for key in keys:
        tensor = reader.get_tensor(key)
        print("name: {}\nshape: {}\ntype:{}".format(key, type(tensor), np.shape(tensor)))
        print("The maximum value of {}: {}".format(key, tensor.max()))
        print("The minimum value of {}: {}\n".format(key, tensor.min()))


if __name__ == '__main__':
    # inspect_ckpt(CHECKPOINT_PATH)
    event_file = os.path.join(TRAIN_DIR, 'events.out.tfevents.1555548474.yq0033-System-Product-Name')
    events = tf.train.summary_iterator(event_file)
    loss = []
    for e in events:
        for v in e.summary.value:
            if 'global_step_1' in v.tag:
                print(v.simple_value)
            if 'learning_rate' in v.tag:
                print(v.simple_value)


