#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import time
import sys

import boto3
import numpy as np
import rasterio as rio
import torch
import torchvision

os.environ['CURL_CA_BUNDLE'] = '/etc/ssl/certs/ca-certificates.crt'

WINDOW_SIZE = 224
MAGIC_NUMBER = (2**16) - 1
OTHER_MAGIC_NUMBER = (2**8) - 1
CHANNELS = 8
MEANS = []
STDS = []


def get_eval_window(raster_ds, mask_ds, x, y):
    window = rio.windows.Window(
        x * WINDOW_SIZE, y * WINDOW_SIZE,
        WINDOW_SIZE, WINDOW_SIZE)

    if CHANNELS == 3:
        bands = [5, 3, 2]
    else:
        bands = raster_ds.indexes[0:CHANNELS]

    # Nodata
    nodata = raster_ds.read(1, window=window) == MAGIC_NUMBER
    not_nodata = (nodata == 0)

    # Labels
    labels = mask_ds.read(1, window=window) == OTHER_MAGIC_NUMBER
    labels = (labels*not_nodata) + (2*nodata)  # class 2 ignored

    # Normalized float32 imagery bands
    data = []
    for band in bands:
        a = raster_ds.read(band, window=window)
        a = np.array((a - MEANS[band-1]) / STDS[band-1], dtype=np.float32)
        a = a * not_nodata
        data.append(a)
    data = np.stack(data, axis=0)

    return (data, labels)


def get_eval_batch(raster_ds, mask_ds, xys, device):
    data = []
    labels = []
    for x, y in xys:
        d, l = get_eval_window(raster_ds, mask_ds, x, y)
        data.append(d)
        labels.append(l)

    data = np.stack(data, axis=0)
    data = torch.from_numpy(data).to(device)
    labels = np.array(np.stack(labels, axis=0), dtype=np.long)
    labels = torch.from_numpy(labels).to(device)
    return (data, labels)


def chunks(l, n):
    # https://chrisalbon.com/python/data_wrangling/break_list_into_chunks_of_equal_size/
    for i in range(0, len(l), n):
        yield l[i:i+n]


if __name__ == "__main__":

    print('ARGUMENTS={}'.format(sys.argv))

    CHANNELS = int(sys.argv[1])
    bucket_name = sys.argv[2]
    mul_name = sys.argv[3]
    mask_name = sys.argv[4]
    model_name = sys.argv[5]
    pred_prefix = sys.argv[6]
    gt_prefix = sys.argv[7]

    print('DATA')

    if not os.path.exists('/tmp/mul.tif'):
        s3 = boto3.client('s3')
        s3.download_file(bucket_name, mul_name, '/tmp/mul.tif')
        del s3
    if not os.path.exists('/tmp/mask.tif'):
        s3 = boto3.client('s3')
        s3.download_file(bucket_name, mask_name, '/tmp/mask.tif')
        del s3

    print('MODEL')

    if not os.path.exists('/tmp/deeplab.pth'):
        s3 = boto3.client('s3')
        s3.download_file(bucket_name, model_name, '/tmp/deeplab.pth')

    print('PRE-COMPUTING')

    with rio.open('/tmp/mul.tif') as raster_ds:
        for i in range(0, len(raster_ds.indexes)):
            a = raster_ds.read(i+1).flatten()
            a = np.extract(a < MAGIC_NUMBER, a)
            MEANS.append(a.mean())
            STDS.append(a.std())
        del a
    print(MEANS)
    print(STDS)

    print('INITIALIZING')

    device = torch.device('cuda')
    deeplab = torch.load('/tmp/deeplab.pth').to(device)
    deeplab.eval()

    print('COMPUTING')

    batch_size = 64
    tps = [0.0, 0.0]
    fps = [0.0, 0.0]
    fns = [0.0, 0.0]
    preds = []
    ground_truth = []

    with rio.open('/tmp/mul.tif') as raster_ds, rio.open('/tmp/mask.tif') as mask_ds, torch.no_grad():

        width = raster_ds.width
        height = raster_ds.height

        xys = []
        for x in range(0, width//WINDOW_SIZE):
            for y in range(0, height//WINDOW_SIZE):
                if ((x + y) % 7 == 0):
                    xy = (x, y)
                    xys.append(xy)

        for xy in chunks(xys, batch_size):
            batch, labels = get_eval_batch(raster_ds, mask_ds, xy, device)
            labels = labels.data.cpu().numpy()
            out = deeplab(batch)['out'].data.cpu().numpy()
            out = np.apply_along_axis(np.argmax, 1, out)
            dont_care = labels == 2
            out = out + 10*dont_care

            for i in range(0, 2):
                tps[i] = tps[i] + ((out == i)*(labels == i)).sum()
                fps[i] = fps[i] + ((out == i)*(labels != i)).sum()
                fns[i] = fns[i] + ((out != i)*(labels == i)).sum()

            preds.append(out.flatten())
            ground_truth.append(labels.flatten())

    print('True Positives  {}'.format(tps))
    print('False Positives {}'.format(fps))
    print('False Negatives {}'.format(fns))

    recalls = []
    precisions = []
    for i in range(0, 2):
        recall = tps[i] / (tps[i] + fns[i])
        recalls.append(recall)
        precision = tps[i] / (tps[i] + fps[i])
        precisions.append(precision)

    print('Recalls    {}'.format(recalls))
    print('Precisions {}'.format(precisions))

    f1s = []
    for i in range(0, 2):
        f1 = 2 * (precisions[i] * recalls[i]) / (precisions[i] + recalls[i])
        f1s.append(f1)
    print('f1 {}'.format(f1s))

    preds = np.concatenate(preds).flatten()
    ground_truth = np.concatenate(ground_truth).flatten()
    preds = np.extract(ground_truth < 2, preds)
    ground_truth = np.extract(ground_truth < 2, ground_truth)
    np.save('/tmp/x.npy', preds, False)
    np.save('/tmp/y.npy', ground_truth, False)
    s3 = boto3.client('s3')
    s3.upload_file('/tmp/x.npy', bucket_name, pred_prefix)
    s3.upload_file('/tmp/y.npy', bucket_name, gt_prefix)
    del s3

# ./download_run.sh s3://raster-vision-mcclain/spacenet/spacenet_eval.py 8 raster-vision-mcclain vegas/data/MUL_AOI_2_Vegas.tif vegas/data/mask_AOI_2_Vegas.tif vegas/deeplab_8channels5x.pth vegas/8channels5x.npy vegas/gt8.npy
# ./download_run.sh s3://raster-vision-mcclain/spacenet/spacenet_eval.py 8 raster-vision-mcclain vegas/data/MUL_AOI_2_Vegas.tif vegas/data/mask_AOI_2_Vegas.tif vegas/deeplab_8channels.pth vegas/8channels.npy vegas/gt8.npy
# ./download_run.sh s3://raster-vision-mcclain/spacenet/spacenet_eval.py 3 raster-vision-mcclain vegas/data/MUL_AOI_2_Vegas.tif vegas/data/mask_AOI_2_Vegas.tif vegas/deeplab_3channels5x.pth vegas/3channels5x.npy vegas/gt3.npy
# ./download_run.sh s3://raster-vision-mcclain/spacenet/spacenet_eval.py 3 raster-vision-mcclain vegas/data/MUL_AOI_2_Vegas.tif vegas/data/mask_AOI_2_Vegas.tif vegas/deeplab_3channels.pth vegas/3channels.npy vegas/gt3.npy
# ./download_run.sh s3://raster-vision-mcclain/spacenet/spacenet_eval.py 8 raster-vision-mcclain vegas_roads/data/MUL_AOI_2_Vegas.tif vegas_roads/data/mask_AOI_2_Vegas.tif vegas_roads/deeplab_8channels.pth vegas_roads/8channels.npy vegas/gt.npy
# ./download_run.sh s3://raster-vision-mcclain/spacenet/spacenet_eval.py 3 raster-vision-mcclain vegas_roads/data/MUL_AOI_2_Vegas.tif vegas_roads/data/mask_AOI_2_Vegas.tif vegas_roads/deeplab_3channels.pth vegas_roads/3channels.npy vegas/gt.npy
