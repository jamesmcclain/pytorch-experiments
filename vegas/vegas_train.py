import os
import time

from PIL import Image

import boto3
import numpy as np
import rasterio as rio
import torch
import torchvision

os.environ['CURL_CA_BUNDLE'] = '/etc/ssl/certs/ca-certificates.crt'

IMAGE_SIZE_X = 14507
IMAGE_SIZE_Y = 13040
SWATCH_SIZE = 224
BATCH_SIZE = 16

if __name__ == "__main__":
    np.random.seed(seed=33)

    device = torch.device('cuda')

    deeplab = torchvision.models.segmentation.deeplabv3_resnet101(
        pretrained=True).to(device)

    raster_ds = rio.open(
        's3://raster-vision-mcclain/vegas/data/MUL_AOI_2_Vegas.tif')
    mask_ds = rio.open(
        's3://raster-vision-mcclain/vegas/data/mask_AOI_2_Vegas.tif')

    last_class = deeplab.classifier[4] = torch.nn.Conv2d(256, 2, kernel_size=(1,1), stride=(1,1)).to(device)
    last_class_aux = deeplab.aux_classifier[4] = torch.nn.Conv2d(256, 2, kernel_size=(1,1), stride=(1,1)).to(device)
    input_filters = torch.nn.Conv2d(8, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False).to(device)