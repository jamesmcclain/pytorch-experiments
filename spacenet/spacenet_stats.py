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

os.environ['CURL_CA_BUNDLE'] = '/etc/ssl/certs/ca-certificates.crt'

MAGIC_NUMBER = (2**16) - 1


def get_normalized_band(raster_ds, i, b = None):
    a = raster_ds.read(i).flatten()
    if b is None:
        a = np.extract(a < MAGIC_NUMBER, a)
    else:
        a = np.extract((a < MAGIC_NUMBER) * b, a)
    mean = a.mean()
    std = a.std()
    a = (a - mean) / std
    return (mean, std, a)


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
        means = []
        stds = []
        bands2 = []
        means2 = []
        stds2 = []
        for i in raster_ds.indexes:
            print('\t BAND {}'.format(i))
            mean, std, a = get_normalized_band(raster_ds, i)
            means.append(mean)
            stds.append(std)
            bands.append(a)
            mean, std, a = get_normalized_band(raster_ds, i, labels == 255)
            means2.append(mean)
            stds2.append(std)
            bands2.append(a)

    means = np.array(means)
    stds = np.array(stds)
    means2 = np.array(means2)
    stds2 = np.array(stds)

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

# ./download_run.sh s3://raster-vision-mcclain/spacenet/spacenet_stats.py
