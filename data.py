# encoding: utf-8

import os, sys, random
import numpy as np

def convertlabels(labels):
    d = np.zeros([len(labels), 10])
    for i in range(len(labels)):
        d[i][labels[i]] = 1
    return d

def train_data():
    with open('train.csv') as f:
        lines = f.readlines()
        data = [[int(x) for x in line.strip().split(',')] for line in lines[1:]]
    TRAIN_SIZE = int(len(data)*0.9)
    random.shuffle(data)
    train_imgs = [d[1:] for d in data[:TRAIN_SIZE]]
    train_labels = [d[0] for d in data[:TRAIN_SIZE]]
    test_imgs = [d[1:] for d in data[TRAIN_SIZE:]]
    test_labels = [d[0] for d in data[TRAIN_SIZE:]]
    return {'train_imgs': np.array(train_imgs, 'float'), 'train_labels': convertlabels(train_labels), 'test_imgs': np.array(test_imgs, 'float'), 'test_labels': convertlabels(test_labels)}

def test_data():
    with open('test.csv') as f:
        lines = f.readlines()
        data = [[int(x) for x in line.strip().split(',')] for line in lines[1:]]
    return {'imgs': np.array(data, 'float')}