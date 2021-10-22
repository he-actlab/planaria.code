import argparse
import logging
from collections import OrderedDict

# from simulator import Simulator, ceil_a_by_b, log2

# Tetris
from nn_dataflow import Network
from nn_dataflow import InputLayer, ConvLayer, FCLayer, PoolingLayer
from nn_dataflow.Layer import DWConvLayer

import os




benchlist = [\
              'GoogleNet', \
              'RESNET-50', \
              'MobileNet-v1',\
              'Tiny-YOLO',\
              'SSD-ResNet-34',\
              'SSD-MobileNet-v1',\
              'GNMT',\
              'YOLOv3'\
            ]


def get_bench_nn(bench_name):
    if bench_name == 'RESNET-50':
        return get_resnet_50()
    elif bench_name == 'GoogleNet':
        return get_googlenet()
    elif bench_name == 'MobileNet-v1':
        return get_mobilenet_v1()
    elif bench_name == 'SSD-ResNet-34':
           return get_ssd_resnet_34()
    elif bench_name == 'SSD-MobileNet-v1':
           return get_ssd_mobilenet_v1()
    elif bench_name == 'GNMT':
           return get_gnmt_encoder(), get_gnmt_decoder()
    elif bench_name == 'Tiny-YOLO':
        return get_yolo()
    elif bench_name == 'YOLOv3':
        return get_yolo_v3()


def write_to_csv(csv_name, fields, stats, network, csv_path='./'):
    if not os.path.exists(csv_path):
        os.makedirs(csv_path)

    for l in stats:
        print(l)
        print(stats[l]['total'])

    bench_csv_name = os.path.join(csv_path, csv_name)
    with open(bench_csv_name, 'w') as f:
        f.write(', '.join(fields+['\n']))
        for l in network:
            if isinstance(network[l], ConvLayer):
                f.write('{}, {}\n'.format(l, ', '.join(str(x) for x in stats[l]['total'])))

def get_bench_numbers(nn, sim_obj, batch_size=1):
    stats = OrderedDict({})
    for layer_name in nn.layer_dict:
        if layer_name != nn.INPUT_LAYER_KEY:
            layer = nn.layer_dict[layer_name]
            #sim_obj.Search(layer)
            out = sim_obj.get_cycles(layer, batch_size)
            if out is not None:
                s, d, cmx, ex_p_c = out
                stats[layer_name] = [s, d, cmx, ex_p_c]
    return stats




def get_mobilenet_v1():
    '''
    MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications
    '''
    NN = Network('MobileNet-v1')

    NN.set_input(InputLayer(3, 224))

    NN.add('conv1', ConvLayer(3, 32, 112, 3, strd =2, iprec=16, wprec=16),
           prevs=(NN.INPUT_LAYER_KEY,))
    NN.add('conv2-dw', DWConvLayer(32, 32, 112, 3, strd=1, iprec=16, wprec=16),
               prevs=('conv1',))
    NN.add('conv3', ConvLayer(32, 64, 112, 1, iprec=16, wprec=16),
           prevs=('conv2-dw',))
    NN.add('conv4-dw', DWConvLayer(64, 64, 56, 3, strd=2, iprec=16, wprec=16),
                prevs=('conv3',))
    NN.add('conv5', ConvLayer(64, 128, 56, 1, iprec=16, wprec=16),
           prevs=('conv4-dw',))
    NN.add('conv6-dw', DWConvLayer(128, 128, 56, 3, strd=1, iprec=16, wprec=16),
                prevs=('conv5',))
    NN.add('conv7', ConvLayer(128, 128, 56, 1, iprec=16, wprec=16),
           prevs=('conv6-dw',))
    NN.add('conv8-dw', DWConvLayer(128, 128, 28, 3, strd=2, iprec=16, wprec=16),
                prevs=('conv7',))
    NN.add('conv9', ConvLayer(128, 256, 28, 1, iprec=16, wprec=16),
           prevs=('conv8-dw',))
    NN.add('conv10-dw', DWConvLayer(256, 256, 28, 3, strd=1, iprec=16, wprec=16),
                prevs=('conv9',))
    NN.add('conv11', ConvLayer(256, 256, 28, 1, iprec=16, wprec=16),
           prevs=('conv10-dw',))
    NN.add('conv12-dw', DWConvLayer(256, 256, 14, 3, strd=2, iprec=16, wprec=16),
                prevs=('conv11',))
    NN.add('conv13', ConvLayer(256, 512, 14, 1, iprec=16, wprec=16),
           prevs=('conv12-dw',))

    NN.add('conv14-dw', DWConvLayer(512, 512, 14, 3, strd=1, iprec=16, wprec=16),
                prevs=('conv13',))
    NN.add('conv15', ConvLayer(512, 512, 14, 1, iprec=16, wprec=16),
           prevs=('conv14-dw',))
    NN.add('conv16-dw', DWConvLayer(512, 512, 14, 3, strd=1, iprec=16, wprec=16),
                prevs=('conv15',))
    NN.add('conv17', ConvLayer(512, 512, 14, 1, iprec=16, wprec=16),
           prevs=('conv16-dw',))
    NN.add('conv18-dw', DWConvLayer(512, 512, 14, 3, strd=1, iprec=16, wprec=16),
                prevs=('conv17',))
    NN.add('conv19', ConvLayer(512, 512, 14, 1, iprec=16, wprec=16),
           prevs=('conv18-dw',))
    NN.add('conv20-dw', DWConvLayer(512, 512, 14, 3, strd=1, iprec=16, wprec=16),
                prevs=('conv19',))
    NN.add('conv21', ConvLayer(512, 512, 14, 1, iprec=16, wprec=16),
           prevs=('conv20-dw',))
    NN.add('conv22-dw', DWConvLayer(512, 512, 14, 3, strd=1, iprec=16, wprec=16),
                prevs=('conv21',))
    NN.add('conv23', ConvLayer(512, 512, 14, 1, iprec=16, wprec=16),
           prevs=('conv22-dw',))
    NN.add('conv24-dw', DWConvLayer(512, 512, 7, 3, strd=2, iprec=16, wprec=16),
                prevs=('conv23',))
    NN.add('conv25', ConvLayer(512, 1024, 7, 1, iprec=16, wprec=16),
           prevs=('conv24-dw',))
    NN.add('conv26-dw', DWConvLayer(1024, 1024, 7, 3, strd=1, iprec=16, wprec=16),
                prevs=('conv25',))
    NN.add('conv27', ConvLayer(1024, 1024, 7, 1, iprec=16, wprec=16),
           prevs=('conv26-dw',))
    NN.add('pool28', PoolingLayer(1024, 1, 7, strd=1), prevs=('conv27',))
    NN.add('fc29', FCLayer(1024, 100, 1, iprec=16, wprec=16))

    return NN


def get_ssd_resnet_34():
    '''
    SSD-ResNet-34 as backbone
    Taken from :
    Fast single shot multibox detector and its application on vehicle counting system

    https://github.com/itayhubara/MLPerf-SSD-R34.pytorch/blob/master/ssd_r34_2m/ssd_r34.py
    '''

    NN = Network('SSD-ResNet-34')

    NN.set_input(InputLayer(3, 320))

    _PREVS = None

    NN.add('conv1', ConvLayer(3, 64, 160, 7, 2, iprec=16, wprec=16))
    NN.add('pool1', PoolingLayer(64, 80, 2))

    for i in range(1, 3):
        NN.add('conv2_{}_a'.format(i),
               ConvLayer(64, 64, 80, 3, iprec=16, wprec=16),
               prevs=_PREVS)
        NN.add('conv2_{}_b'.format(i), ConvLayer(64, 64, 80, 3, iprec=16, wprec=16))
        # NN.add('conv2_{}_c'.format(i), ConvLayer(64, 256, 56, 1))

        # With residual shortcut.
        if i == 1:
            _PREVS = None
        else:
            _PREVS = ('conv2_{}_b'.format(i), NN.prev_layers('conv2_{}_a'.format(i))[0][0])

    for i in range(1, 5):
        NN.add('conv3_{}_a'.format(i),
               ConvLayer(64, 128, 40, 3, 2, iprec=16, wprec=16) if i == 1
               else ConvLayer(128, 128, 40, 3, iprec=16, wprec=16),
               prevs=_PREVS)
        NN.add('conv3_{}_b'.format(i), ConvLayer(128, 128, 40, 3, iprec=16, wprec=16))

        # With residual shortcut.
        if i == 1:
            # Residual does not cross module.
            _PREVS = None
        else:
            _PREVS = ('conv3_{}_b'.format(i), 'conv3_{}_b'.format(i - 1))

    NN.add('fc6_a', ConvLayer(128, 256, 40, 3, iprec=16, wprec=16), prevs=_PREVS)
    NN.add('fc6_b', ConvLayer(256, 512, 20, 3, 2, iprec=16, wprec=16), prevs=('fc6_a',))

    NN.add('fc7', ConvLayer(512, 512, 20, 1, iprec=16, wprec=16), prevs=('fc6_b',))
    NN.add('conv6_1', ConvLayer(512, 128, 20, 1, iprec=16, wprec=16), prevs=('fc7',))
    NN.add('conv6_2', ConvLayer(128, 256, 10, 3, 2, iprec=16, wprec=16), prevs=('conv6_1',))
    NN.add('conv7_1', ConvLayer(256, 64, 10, 1, iprec=16, wprec=16), prevs=('conv6_2',))
    NN.add('conv7_2', ConvLayer(64, 128, 5, 3, 2, iprec=16, wprec=16), prevs=('conv7_1',))
    NN.add('conv8_1', ConvLayer(128, 64, 5, 1, iprec=16, wprec=16), prevs=('conv7_2',))
    NN.add('conv8_2', ConvLayer(64, 128, 3, 3, 1, iprec=16, wprec=16), prevs=('conv8_1',))
    NN.add('conv9_1', ConvLayer(128, 64, 3, 1, iprec=16, wprec=16), prevs=('conv8_2',))
    NN.add('conv9_2', ConvLayer(64, 128, 1, 3, 2, iprec=16, wprec=16), prevs=('conv9_1',))

    return NN


def get_ssd_mobilenet_v1():
    '''
    SSD MobileNet-v1
    Model taken from:
    https://seongkyun.github.io/study/2019/04/24/ssds_architecture/
    '''

    NN = Network('ssd_mobilent_v1')


    NN.set_input(InputLayer(3, 320))

    NN.add('conv1', ConvLayer(3, 32, 160, 3, strd =2, iprec=16, wprec=16),
           prevs=(NN.INPUT_LAYER_KEY,))
    NN.add('conv2-dw', DWConvLayer(32, 32, 160, 3, strd=1, iprec=16, wprec=16),
               prevs=('conv1',))
    NN.add('conv3', ConvLayer(32, 64, 160, 1, iprec=16, wprec=16),
           prevs=('conv2-dw',))
    NN.add('conv4-dw', DWConvLayer(64, 64, 80, 3, strd=2, iprec=16, wprec=16),
                prevs=('conv3',))
    NN.add('conv5', ConvLayer(64, 128, 80, 1, iprec=16, wprec=16),
           prevs=('conv4-dw',))
    NN.add('conv6-dw', DWConvLayer(128, 128, 80, 3, strd=1, iprec=16, wprec=16),
                prevs=('conv5',))
    NN.add('conv7', ConvLayer(128, 128, 80, 1, iprec=16, wprec=16),
           prevs=('conv6-dw',))
    NN.add('conv8-dw', DWConvLayer(128, 128, 40, 3, strd=2, iprec=16, wprec=16),
                prevs=('conv7',))
    NN.add('conv9', ConvLayer(128, 256, 40, 1, iprec=16, wprec=16),
           prevs=('conv8-dw',))
    NN.add('conv10-dw', DWConvLayer(256, 256, 40, 3, strd=1, iprec=16, wprec=16),
                prevs=('conv9',))
    NN.add('conv11', ConvLayer(256, 256, 40, 1, iprec=16, wprec=16),
           prevs=('conv10-dw',))
    NN.add('conv12-dw', DWConvLayer(256, 256, 20, 3, strd=2, iprec=16, wprec=16),
                prevs=('conv11',))
    NN.add('conv13', ConvLayer(256, 512, 20, 1, iprec=16, wprec=16),
           prevs=('conv12-dw',))

    NN.add('conv-fc6', ConvLayer(512, 1024, 10, 3, 2, iprec=16, wprec=16), prevs=('conv13',))
    NN.add('conv-fc7', ConvLayer(1024, 1024, 10, 1, 1, iprec=16, wprec=16), prevs=('conv-fc6',))
    NN.add('conv14_1', ConvLayer(1024, 256, 10, 1, 1, iprec=16, wprec=16), prevs=('conv-fc7',))
    NN.add('conv14_2', ConvLayer(256, 512, 5, 3, 2, iprec=16, wprec=16), prevs=('conv14_1',))
    NN.add('conv15_1', ConvLayer(512, 128, 5, 1, 1, iprec=16, wprec=16), prevs=('conv14_2',))
    NN.add('conv15_2', ConvLayer(128, 256, 3, 3, 1, iprec=16, wprec=16), prevs=('conv15_1',))
    NN.add('conv16_1', ConvLayer(256, 128, 3, 1, 1, iprec=16, wprec=16), prevs=('conv15_2',))
    NN.add('conv16_2', ConvLayer(128, 256, 2, 3, 1, iprec=16, wprec=16), prevs=('conv16_1',))
    NN.add('conv17_1', ConvLayer(256, 64, 2, 1, 1, iprec=16, wprec=16), prevs=('conv16_2',))
    NN.add('conv17_2', ConvLayer(64, 128, 1, 3, 1, iprec=16, wprec=16), prevs=('conv17_1',))

    return NN




def get_gnmt_encoder():
    '''
    topology from NN distiller
    '''

    NN = Network('GNMT-Encoder')
    #### This the base network
    NN.set_input(InputLayer(512, 1))

    NN.add('gnmt-en-fc1', FCLayer(512, 512, 1, iprec=16, wprec=16), prevs=(NN.INPUT_LAYER_KEY))
    NN.add('gnmt-en-fc2', FCLayer(512, 2*512, 1, iprec=16, wprec=16), prevs=('gnmt-en-fc1',))
    NN.add('gnmt-en-fc3', FCLayer(2*512, 512, 1, iprec=16, wprec=16), prevs=('gnmt-en-fc2',))
    NN.add('gnmt-en-fc4', FCLayer(512, 512, 1, iprec=16, wprec=16), prevs=('gnmt-en-fc3',))
    NN.add('gnmt-en-fc5', FCLayer(512, 512, 1, iprec=16, wprec=16), prevs=('gnmt-en-fc4',))
    NN.add('gnmt-en-fc6', FCLayer(512, 512, 1, iprec=16, wprec=16), prevs=('gnmt-en-fc5',))
    NN.add('gnmt-en-fc7', FCLayer(512, 512, 1, iprec=16, wprec=16), prevs=('gnmt-en-fc6',))
    NN.add('gnmt-en-fc8', FCLayer(512, 512, 1, iprec=16, wprec=16), prevs=('gnmt-en-fc7',))

    return NN

def get_gnmt_decoder():
    '''
    topology from NN distiller
    '''

    NN = Network('GNMT-Decoder')
    #### This the base network
    NN.set_input(InputLayer(512, 1))

    NN.add('gnmt-en-fc1', FCLayer(512, 512, 1, iprec=16, wprec=16), prevs=(NN.INPUT_LAYER_KEY))

    NN.add('gnmt-en-fc2', FCLayer(512, 2*512, 1, iprec=16, wprec=16), prevs=('gnmt-en-fc1',))
    NN.add('gnmt-en-fc3', FCLayer(2*512, 2*512, 1, iprec=16, wprec=16), prevs=('gnmt-en-fc2',))
    NN.add('gnmt-en-fc4', FCLayer(2*512, 2*512, 1, iprec=16, wprec=16), prevs=('gnmt-en-fc3',))
    NN.add('gnmt-en-fc5', FCLayer(2*512, 2*512, 1, iprec=16, wprec=16), prevs=('gnmt-en-fc4',))
    NN.add('gnmt-en-fc6', FCLayer(2*512, 2*512, 1, iprec=16, wprec=16), prevs=('gnmt-en-fc5',))
    NN.add('gnmt-en-fc7', FCLayer(2*512, 2*512, 1, iprec=16, wprec=16), prevs=('gnmt-en-fc6',))
    NN.add('gnmt-en-fc8', FCLayer(2*512, 2*512, 1, iprec=16, wprec=16), prevs=('gnmt-en-fc7',))

    return NN



def get_yolo():


    NN = Network('YOLO_v2_tiny')

    NN.set_input(InputLayer(3, 416))

    NN.add('conv0', ConvLayer(3, 16, 416, 3, 1, iprec=8, wprec=8),
       prevs=(NN.INPUT_LAYER_KEY,))
    NN.add('pool0', PoolingLayer(16, 208, 2), prevs=('conv0',))

    NN.add('conv1', ConvLayer(16, 32, 208, 3, 1, iprec=8, wprec=8),
           prevs=('pool0'))
    NN.add('pool1', PoolingLayer(32, 104, 2), prevs=('conv1',))
    NN.add('conv2', ConvLayer(32, 64, 104, 3, 1, iprec=8, wprec=8),
           prevs=('pool1'))
    NN.add('pool2', PoolingLayer(64, 52, 2), prevs=('conv2',))
    NN.add('conv3', ConvLayer(64, 128, 52, 3, 1, iprec=8, wprec=8),
           prevs=('pool2'))
    NN.add('pool3', PoolingLayer(128, 26, 2), prevs=('conv3',))
    NN.add('conv4', ConvLayer(128, 256, 26, 3, 1, iprec=8, wprec=8),
           prevs=('pool3'))
    NN.add('pool4', PoolingLayer(256, 13, 2), prevs=('conv4',))
    NN.add('conv5', ConvLayer(256, 512, 13, 3, 1, iprec=8, wprec=8),
           prevs=('pool4'))
    #NN.add('pool5', PoolingLayer(512, 13, 2), prevs=('conv5',))

    NN.add('conv6', ConvLayer(512, 1024, 13, 3, 1, iprec=8, wprec=8),
           prevs=('conv5'))
    NN.add('conv7', ConvLayer(1024, 1024, 13, 3, 1, iprec=8, wprec=8),
           prevs=('conv6'))
    NN.add('conv8', ConvLayer(1024, 125, 13, 1, 1, iprec=8, wprec=8),
           prevs=('conv7'))

    return NN






def get_yolo_v3():


    NN = Network('YOLOv3')

    NN.set_input(InputLayer(3, 256))

    _PREVS = None

    NN.add('conv1', ConvLayer(3, 32, 256, 3, 1))
    NN.add('conv2', ConvLayer(32, 64, 128, 3, 2), prevs=('conv1'))

    for i in range(1, 2):
        NN.add('conv3_{}_a'.format(i),
               ConvLayer(64, 32, 128, 1),
               prevs=('conv2'))
        NN.add('conv3_{}_b'.format(i), ConvLayer(32, 64, 128, 3))

        # With residual shortcut.
        if i == 1:
            # Residual does not cross module.
            _PREVS = None
        else:
            _PREVS = ('conv3_{}_b'.format(i), 'conv3_{}_b'.format(i - 1))

    NN.add('conv4', ConvLayer(64, 128, 64, 3, 2))
    for i in range(1, 3):
        NN.add('conv5_{}_a'.format(i),
               ConvLayer(128, 64, 64, 1) if i == 1 else ConvLayer(128, 64, 64, 1),
               prevs=('conv4'))
        NN.add('conv5_{}_b'.format(i), ConvLayer(64, 128, 64, 3))

        # With residual shortcut.
        if i == 1:
            # Residual does not cross module.
            _PREVS = None
        else:
            _PREVS = ('conv5_{}_b'.format(i), 'conv5_{}_b'.format(i - 1))

    NN.add('conv6', ConvLayer(128, 256, 32, 3, 2))

    for i in range(1, 9):
        NN.add('conv7_{}_a'.format(i),
               ConvLayer(256, 128, 32, 1) if i == 1 else ConvLayer(256, 128, 32, 1),
               prevs=('conv6'))
        NN.add('conv7_{}_b'.format(i), ConvLayer(128, 256, 32, 3))

        # With residual shortcut.
        if i == 1:
            # Residual does not cross module.
            _PREVS = None
        else:
            _PREVS = ('conv7_{}_b'.format(i), 'conv7_{}_b'.format(i - 1))

    NN.add('conv8', ConvLayer(256, 512, 16, 3, 2))

    for i in range(1, 9):
        NN.add('conv9_{}_a'.format(i),
               ConvLayer(512, 256, 16, 1) if i == 1 else ConvLayer(512, 256, 16, 1),
               prevs=('conv8'))
        NN.add('conv9_{}_b'.format(i), ConvLayer(256, 512, 16, 3))

        # With residual shortcut.
        if i == 1:
            # Residual does not cross module.
            _PREVS = None
        else:
            _PREVS = ('conv9_{}_b'.format(i), 'conv9_{}_b'.format(i - 1))

    NN.add('conv10', ConvLayer(512, 1024, 8, 3, 2))

    for i in range(1, 5):
            NN.add('conv11_{}_a'.format(i),
                   ConvLayer(1024, 512, 8, 1) if i == 1 else ConvLayer(1024, 512, 8, 1),
                   prevs=('conv10'))
            NN.add('conv11_{}_b'.format(i), ConvLayer(512, 1024, 8, 3))

            # With residual shortcut.
            if i == 1:
                # Residual does not cross module.
                _PREVS = None
            else:
                _PREVS = ('conv11_{}_b'.format(i), 'conv11_{}_b'.format(i - 1))

    return NN



def get_resnet_50():


    NN = Network('ResNet-50')

    NN.set_input(InputLayer(3, 224))

    _PREVS = None

    NN.add('conv1', ConvLayer(3, 64, 112, 7, 2))
    NN.add('pool1', PoolingLayer(64, 56, 2))

    for i in range(1, 4):
        NN.add('conv2_{}_a'.format(i),
               ConvLayer(64, 64, 56, 1) if i == 1 else ConvLayer(256, 64, 56, 1),
               prevs=_PREVS)
        NN.add('conv2_{}_b'.format(i), ConvLayer(64, 64, 56, 3))
        NN.add('conv2_{}_c'.format(i), ConvLayer(64, 256, 56, 1))

        # With residual shortcut.
        if i == 1:
            # Residual does not cross module.
            _PREVS = None
        else:
            _PREVS = ('conv2_{}_c'.format(i), 'conv2_{}_c'.format(i - 1))

    for i in range(1, 5):
        NN.add('conv3_{}_a'.format(i),
               ConvLayer(256, 128, 28, 1, 2) if i == 1
               else ConvLayer(512, 128, 28, 1),
               prevs=_PREVS)
        NN.add('conv3_{}_b'.format(i), ConvLayer(128, 128, 28, 3))
        NN.add('conv3_{}_c'.format(i), ConvLayer(128, 512, 28, 1))

        # With residual shortcut.
        if i == 1:
            # Residual does not cross module.
            _PREVS = None
        else:
            _PREVS = ('conv3_{}_c'.format(i), 'conv3_{}_c'.format(i - 1))

    for i in range(1, 7):
        NN.add('conv4_{}_a'.format(i),
               ConvLayer(512, 256, 14, 1, 2) if i == 1
               else ConvLayer(1024, 256, 14, 1),
               prevs=_PREVS)
        NN.add('conv4_{}_b'.format(i), ConvLayer(256, 256, 14, 3))
        NN.add('conv4_{}_c'.format(i), ConvLayer(256, 1024, 14, 1))

        # With residual shortcut.
        if i == 1:
            # Residual does not cross module.
            _PREVS = None
        else:
            _PREVS = ('conv4_{}_c'.format(i), 'conv4_{}_c'.format(i - 1))

    for i in range(1, 4):
        NN.add('conv5_{}_a'.format(i),
               ConvLayer(1024, 512, 7, 1, 2) if i == 1
               else ConvLayer(2048, 512, 7, 1),
               prevs=_PREVS)
        NN.add('conv5_{}_b'.format(i), ConvLayer(512, 512, 7, 3))
        NN.add('conv5_{}_c'.format(i), ConvLayer(512, 2048, 7, 1))

        # With residual shortcut.
        if i == 1:
            # Residual does not cross module.
            _PREVS = None
        else:
            _PREVS = ('conv5_{}_c'.format(i), 'conv5_{}_c'.format(i - 1))

    return NN


def get_googlenet():


    NN = Network('GoogleNet')

    NN.set_input(InputLayer(3, 224))

    NN.add('conv1', ConvLayer(3, 64, 112, 7, 2, iprec=16, wprec=16))
    NN.add('pool1', PoolingLayer(64, 56, 3, strd=2))
    # Norm layer is ignored.

    NN.add('conv2_3x3_reduce', ConvLayer(64, 64, 56, 1, iprec=16, wprec=16))
    NN.add('conv2_3x3', ConvLayer(64, 192, 56, 3, iprec=16, wprec=16))
    # Norm layer is ignored.
    NN.add('pool2', PoolingLayer(192, 28, 3, strd=2))


    def add_inception(network, incp_id, sfmap, nfmaps_in, nfmaps_1, nfmaps_3r,
                      nfmaps_3, nfmaps_5r, nfmaps_5, nfmaps_pool, prevs):
        ''' Add an inception module to the network. '''
        pfx = 'inception_{}_'.format(incp_id)
        # 1x1 branch.
        network.add(pfx + '1x1', ConvLayer(nfmaps_in, nfmaps_1, sfmap, 1, iprec=16, wprec=16),
                    prevs=prevs)
        # 3x3 branch.
        network.add(pfx + '3x3_reduce', ConvLayer(nfmaps_in, nfmaps_3r, sfmap, 1, iprec=16, wprec=16),
                    prevs=prevs)
        network.add(pfx + '3x3', ConvLayer(nfmaps_3r, nfmaps_3, sfmap, 3, iprec=16, wprec=16))
        # 5x5 branch.
        network.add(pfx + '5x5_reduce', ConvLayer(nfmaps_in, nfmaps_5r, sfmap, 1, iprec=16, wprec=16),
                    prevs=prevs)
        network.add(pfx + '5x5', ConvLayer(nfmaps_5r, nfmaps_5, sfmap, 5, iprec=16, wprec=16))
        # Pooling branch.
        network.add(pfx + 'pool_proj', ConvLayer(nfmaps_in, nfmaps_pool, sfmap, 1),
                    prevs=prevs)
        # Merge branches.
        return (pfx + '1x1', pfx + '3x3', pfx + '5x5', pfx + 'pool_proj')


    _PREVS = ('pool2',)

    # Inception 3.
    _PREVS = add_inception(NN, '3a', 28, 192, 64, 96, 128, 16, 32, 32,
                           prevs=_PREVS)
    _PREVS = add_inception(NN, '3b', 28, 256, 128, 128, 192, 32, 96, 64,
                           prevs=_PREVS)

    NN.add('pool3', PoolingLayer(480, 14, 3, strd=2), prevs=_PREVS)
    _PREVS = ('pool3',)

    # Inception 4.
    _PREVS = add_inception(NN, '4a', 14, 480, 192, 96, 208, 16, 48, 64,
                           prevs=_PREVS)
    _PREVS = add_inception(NN, '4b', 14, 512, 160, 112, 224, 24, 64, 64,
                           prevs=_PREVS)
    _PREVS = add_inception(NN, '4c', 14, 512, 128, 128, 256, 24, 64, 64,
                           prevs=_PREVS)
    _PREVS = add_inception(NN, '4d', 14, 512, 112, 144, 288, 32, 64, 64,
                           prevs=_PREVS)
    _PREVS = add_inception(NN, '4e', 14, 528, 256, 160, 320, 32, 128, 128,
                           prevs=_PREVS)

    NN.add('pool4', PoolingLayer(832, 7, 3, strd=2), prevs=_PREVS)
    _PREVS = ('pool4',)

    # Inception 5.
    _PREVS = add_inception(NN, '5a', 7, 832, 256, 160, 320, 32, 128, 128,
                           prevs=_PREVS)
    _PREVS = add_inception(NN, '5b', 7, 832, 384, 192, 384, 48, 128, 128,
                           prevs=_PREVS)

    NN.add('pool5', PoolingLayer(1024, 1, 7), prevs=_PREVS)

    NN.add('fc', FCLayer(1024, 1000, 1, iprec=16, wprec=16))

    return NN




