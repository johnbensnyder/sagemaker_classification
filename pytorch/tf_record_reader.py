from glob import glob
import sys
import os
import cv2
import imutils

import torch
from tfrecord.torch.dataset import MultiTFRecordDataset

def _create_single_index(input_file):
    output_file = input_file.replace('tfrecord', 'index')
    os.system("python -m tfrecord.tools.tfrecord2idx {0} {1}".format(input_file, output_file)) 
    return

def _create_index(files):
    for file in files:
        if 'tfrecord' in file.split('.')[-1]:
            _create_single_index(file)
    return

def _create_splits(file_pattern):
    files = glob(file_pattern)
    trimmed_file_pattern = file_pattern.replace('*', '')
    prob = 1/len(files)
    splits = {file.replace(trimmed_file_pattern, ''): prob for file in files}
    return splits

def decode_image(features):
    # get BGR image from bytes
    features["image"] = cv2.imdecode(features["image"], -1)
    return features

def flip_channels(features):
    if len(features["image"].shape)==2:
        features["image"] = cv2.cvtColor(features["image"], cv2.COLOR_GRAY2RGB)
    else:
        features["image"] = cv2.cvtColor(features["image"], cv2.COLOR_BGR2RGB)
    return features

def resize_and_pad(features, size=224):
    if features["image"].shape[0]>=features["image"].shape[1]:
        size_kwargs = {'height': size}
    else:
        size_kwargs = {'width': size}
    resized = imutils.resize(features["image"], **size_kwargs)
    height, width = resized.shape[:2]
    bottom = size-height
    right = size-width
    features["image"] = cv2.copyMakeBorder(resized, 0, bottom, 0, right, cv2.BORDER_CONSTANT)
    return features

def preprocess(features):
    features = decode_image(features)
    features = flip_channels(features)
    features = resize_and_pad(features)
    return features

def tf_record_reader(file_pattern, shuffle=None):
    files = glob(file_pattern)
    _create_index(files)
    splits = _create_splits(file_pattern)
    data_pattern = file_pattern.replace('*', '{}')
    index_pattern = file_pattern.replace('tfrecord', 'index').replace('*', '{}')
    description = {"image": "byte", "label": "int"}
    dataset = MultiTFRecordDataset(data_pattern, index_pattern, splits, description,
                               transform=preprocess, shuffle_queue_size=shuffle)
    return dataset
