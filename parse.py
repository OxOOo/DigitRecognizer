# encoding: utf-8

import os, sys
import numpy as np
import cv2

os.system('mkdir -p images/train')
os.system('mkdir -p images/test')

# with open('train.csv') as f:
#     lines = f.readlines()
#     for index in range(1, len(lines)):
#         data = [int(x) for x in lines[index].strip().split(',')]
#         label = data[0]
#         img = np.array(data[1:]).reshape([28, 28])
#         cv2.imwrite('images/train/%d[%d].jpg' % (index, label), img)
#         if index % 100 == 0:
#             print 'index', index

with open('test.csv') as f:
    lines = f.readlines()
    for index in range(1, len(lines)):
        data = [int(x) for x in lines[index].strip().split(',')]
        img = np.array(data[:]).reshape([28, 28])
        cv2.imwrite('images/test/%d.jpg' % (index), img)
        if index % 100 == 0:
            print 'index', index