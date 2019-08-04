#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
import os
import sys
import time

import numpy as np

import boto3
import rasterio as rio
from sklearn.metrics import mutual_info_score
from tabulate import tabulate
from scipy.stats import entropy

os.environ['CURL_CA_BUNDLE'] = '/etc/ssl/certs/ca-certificates.crt'

MAGIC_NUMBER = (2**16) - 1


def get_normalized_band(raster_ds, i, b=None):
    a = raster_ds.read(i).flatten()
    if b is None:
        a = np.extract(a < MAGIC_NUMBER, a)
    else:
        a = np.extract((a < MAGIC_NUMBER) * b, a)
    mean = a.mean()
    std = a.std()
    a = (a - mean) / std
    return a


if __name__ == "__main__":

    print('DATA')

    if len(sys.argv) > 2:
        bucket = sys.argv[2]
    else:
        bucket = 'raster-vision-mcclain'

    if not os.path.exists('/tmp/mul.tif'):
        s3 = boto3.client('s3')
        s3.download_file(bucket, sys.argv[3], '/tmp/mul.tif')
        del s3

    if not os.path.exists('/tmp/mask.tif'):
        s3 = boto3.client('s3')
        s3.download_file(bucket, sys.argv[4], '/tmp/mask.tif')
        del s3

    print('COMPUTING')

    with rio.open('/tmp/mask.tif') as mask_ds:
        labels = mask_ds.read(1).flatten()
    with rio.open('/tmp/mul.tif') as raster_ds:
        bands = []
        bands2 = []
        for i in raster_ds.indexes:
            print('\t BAND {}'.format(i))
            a = get_normalized_band(raster_ds, i)
            bands.append(a)
            a = get_normalized_band(raster_ds, i, labels == 255)
            bands2.append(a)
        band1 = raster_ds.read(i).flatten()
        labels = np.extract(band1 < MAGIC_NUMBER, labels == 255)
        del band1

    mi_tab = np.identity(len(bands))
    mi_tab2 = np.identity(len(bands))
    for i in range(0, len(bands)):
        for j in range(i+1, len(bands)):
            c_xy = np.histogram2d(bands[i], bands[j])[0]
            mi_tab[i][j] = mi_tab[j][i] = mutual_info_score(None, None, c_xy)
            c_xy = np.histogram2d(bands2[i], bands2[j])[0]
            mi_tab2[i][j] = mi_tab2[j][i] = mutual_info_score(None, None, c_xy)
    print(tabulate(mi_tab, tablefmt='fancy_grid'))
    print(tabulate(mi_tab2, tablefmt='fancy_grid'))

    mi_tab3 = np.zeros((len(bands), 1))
    for i in range(0, len(bands)):
        c_xy = np.histogram2d(bands[i], labels)[0]
        mi_tab3[i][0] = mutual_info_score(None, None, c_xy)
    print(tabulate(mi_tab3, tablefmt='fancy_grid'))

    entropy_tab = np.zeros((len(bands)+1, 1))
    for i in range(0, len(bands)):
        hist = np.histogram(bands[i])[0]
        entropy_tab[i][0] = entropy(hist)
    entropy_tab[len(bands)][0] = entropy(np.histogram(labels)[0])
    print(tabulate(entropy_tab, tablefmt='fancy_grid'))

# ./download_run.sh s3://raster-vision-mcclain/spacenet/spacenet_stats.py
