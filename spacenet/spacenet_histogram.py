import os
import sys
import time
import math

import numpy as np

import boto3
import rasterio as rio

os.environ['CURL_CA_BUNDLE'] = '/etc/ssl/certs/ca-certificates.crt'

IMAGE_SIZE_X = 14507
IMAGE_SIZE_Y = 13040
SWATCH_SIZE = 224
BATCH_SIZE = 16
MAGIC_NUMBER = (2**16) - 1

if __name__ == "__main__":

    print('DATA')
    if not os.path.exists('/tmp/mul.tif'):
        s3 = boto3.client('s3')
        s3.download_file('raster-vision-mcclain', 'vegas/data/MUL_AOI_2_Vegas.tif', '/tmp/mul.tif')
        del s3

    if not os.path.exists('/tmp/mask.tif'):
        s3 = boto3.client('s3')
        s3.download_file('raster-vision-mcclain',
                         'vegas/data/mask_AOI_2_Vegas.tif', '/tmp/mask.tif')
        del s3

    with rio.open('/tmp/mul.tif') as raster_ds, rio.open('/tmp/mask.tif') as mask_ds:
        print('COMPUTING')
        for i in raster_ds.indexes:
            band = raster_ds.read(i)
            good = (band != MAGIC_NUMBER)
            band = (band * good)
            mu = float(band.sum()) / float(good.sum())
            band = band - mu
            band = (band * band)
            sigma = math.sqrt(float(band.sum()) / float(good.sum()))

            with open('output.txt', 'a') as f:
                f.write('BAND={} mu={} sigma={}'.format(i, mu, sigma))
                print('BAND={} mu={} sigma={}'.format(i, mu, sigma))

# ./download_run_upload.sh s3://raster-vision-mcclain/vegas/vegas_histogram.py output.txt s3://raster-vision-mcclain/vegas/vegas_histogram.txt
