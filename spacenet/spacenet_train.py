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
MEAN = None
STD = None

# BAND=1 mu=282.25613565591533 sigma=276.51741601066635
# BAND=2 mu=434.92568434644414 sigma=449.7047682214329
# BAND=3 mu=576.778925128917 sigma=621.334564553244
# BAND=4 mu=431.0893725326917 sigma=482.97355488737406
# BAND=5 mu=413.849793435062 sigma=475.73731167726834
# BAND=6 mu=407.8019863596957 sigma=446.7792042730571
# BAND=7 mu=540.9877607404478 sigma=595.4577943508493
# BAND=8 mu=346.13134404092824 sigma=381.1634917763915


def get_random_training_window(raster_ds, mask_ds, width, height):
    x2 = 0
    y2 = 0
    while ((x2 + y2) % 7) == 0:
        x = np.random.randint(0, width - WINDOW_SIZE)
        y = np.random.randint(0, height - WINDOW_SIZE)
        x2 = int(x / WINDOW_SIZE)
        y2 = int(y / WINDOW_SIZE)
    window = rio.windows.Window(
        x2 * WINDOW_SIZE, y2 * WINDOW_SIZE,
        WINDOW_SIZE, WINDOW_SIZE)

    data = []
    if CHANNELS == 3:
        bands = [5, 3, 2]
    else:
        bands = raster_ds.indexes[0:8]
    for i in bands:
        a = raster_ds.read(i, window=window)
        data.append(a)

    labels = mask_ds.read(1, window=window)
    labels = labels / OTHER_MAGIC_NUMBER
    nodata = (data[0] == MAGIC_NUMBER)
    not_nodata = (data[0] != MAGIC_NUMBER)
    labels = (labels * not_nodata) + (2*nodata)  # class 2 ignored

    # Normalize
    for i in range(0, len(bands)):
        data[i] = np.array(data[i] / float((2**16) -1), dtype=np.float32)
        if MEAN and STD:
            data[i] = data[i] - MEAN[i]
            data[i] = data[i] / STD[i]

    data = np.stack(data, axis=0)
    return (data, labels)


def get_random_training_batch(raster_ds, mask_ds, width, height, batch_size, device):
    data = []
    labels = []
    for i in range(0, batch_size):
        d, l = get_random_training_window(raster_ds, mask_ds, width, height)
        data.append(d)
        labels.append(l)

    data = np.stack(data, axis=0)
    data = torch.from_numpy(data).to(device)
    labels = np.array(np.stack(labels, axis=0), dtype=np.long)
    labels = torch.from_numpy(labels).to(device)
    return (data, labels)


def train(model, opt, obj, steps_per_epoch, epochs, batch_size, raster_ds, mask_ds, width, height, device):
    model.train()
    current_time = time.time()
    for i in range(epochs):
        avg_loss = 0.0
        for j in range(steps_per_epoch):
            batch_tensor = get_random_training_batch(
                raster_ds, mask_ds, width, height, batch_size, device)
            opt.zero_grad()
            pred = model(batch_tensor[0])
            loss = 1.0*obj(pred.get('out'), batch_tensor[1]) \
                + 0.4*obj(pred.get('aux'), batch_tensor[1])
            loss.backward()
            opt.step()
            avg_loss = avg_loss + loss.item()
        avg_loss = avg_loss / steps_per_epoch
        last_time = current_time
        current_time = time.time()
        print('\t\t epoch={} time={} avg_loss={}'.format(
            i, current_time - last_time, avg_loss))


if __name__ == "__main__":

    if len(sys.argv) > 0:
        CHANNELS = int(sys.argv[1])
    print('CHANNELS={}'.format(CHANNELS))

    if len(sys.argv) > 2:
        epochs = int(sys.argv[2])
    else:
        epochs = 10
    print('EPOCHS PER NUGGET={}'.format(epochs))

    if len(sys.argv) > 5:
        mul_name = sys.argv[3]
        mask_name = sys.argv[4]
        dataset_name = sys.argv[5]
    else:
        mul_name = 'vegas/data/MUL_AOI_2_Vegas.tif'
        mask_name = 'vegas/data/mask_AOI_2_Vegas.tif'
        dataset_name = 'vegas'
    print('{} {} {}'.format(mul_name, mask_name, dataset_name))

    if len(sys.argv) > 5 + 2*CHANNELS:
        MEAN = []
        STD = []
        for i in range(6,6+CHANNELS):
            MEAN.append(float(sys.argv[i]))
            STD.append(float(sys.argv[i + CHANNELS]))

    print('DATA')
    if not os.path.exists('/tmp/mul.tif'):
        s3 = boto3.client('s3')
        s3.download_file('raster-vision-mcclain', mul_name, '/tmp/mul.tif')
        del s3
    if not os.path.exists('/tmp/mask.tif'):
        s3 = boto3.client('s3')
        s3.download_file('raster-vision-mcclain', mask_name, '/tmp/mask.tif')
        del s3

    print('INITIALIZING')
    np.random.seed(seed=33)
    device = torch.device('cuda')
    deeplab = torchvision.models.segmentation.deeplabv3_resnet101(
        pretrained=True).to(device)
    last_class = deeplab.classifier[4] = torch.nn.Conv2d(
        256, 2, kernel_size=(1, 1), stride=(1, 1)).to(device)
    last_class_aux = deeplab.aux_classifier[4] = torch.nn.Conv2d(
        256, 2, kernel_size=(1, 1), stride=(1, 1)).to(device)
    input_filters = deeplab.backbone.conv1 = torch.nn.Conv2d(CHANNELS, 64, kernel_size=(
        7, 7), stride=(2, 2), padding=(3, 3), bias=False).to(device)

    print('COMPUTING')
    with rio.open('/tmp/mul.tif') as raster_ds, rio.open('/tmp/mask.tif') as mask_ds:

        if (raster_ds.height != mask_ds.height) or (raster_ds.width != mask_ds.width):
            print('PROBLEM WITH DIMENSIONS')
            sys.exit()

        width = raster_ds.width
        height = raster_ds.height
        batch_size = 16
        steps_per_epoch = int((width * height * 6.0) /
                              (WINDOW_SIZE * WINDOW_SIZE * 7.0 * batch_size))
        print('\t STEPS PER EPOCH={}'.format(steps_per_epoch))

        obj = torch.nn.CrossEntropyLoss(ignore_index=2).to(device)

        print('\t TRAINING FIRST AND LAST LAYERS')
        try:
            s3 = boto3.client('s3')
            s3.download_file('raster-vision-mcclain', '{}/deeplab_first_and_last_{}epochs_{}channels_1.pth'.format(
                dataset_name, epochs, CHANNELS), 'deeplab.pth')
            deeplab = torch.load('deeplab.pth').to(device)
            print('\t\t SUCCESSFULLY RESTARTED')
        except:
            for p in deeplab.parameters():
                p.requires_grad = False
            for p in last_class.parameters():
                p.requires_grad = True
            for p in last_class_aux.parameters():
                p.requires_grad = True
            for p in input_filters.parameters():
                p.requires_grad = True

            ps = []
            for n, p in deeplab.named_parameters():
                if p.requires_grad == True:
                    ps.append(p)
                else:
                    p.grad = None
            opt = torch.optim.SGD(ps, lr=0.01, momentum=0.9)

            train(deeplab, opt, obj, steps_per_epoch, epochs // 2, batch_size,
                  raster_ds, mask_ds, width, height, device)

            print('\t UPLOADING')
            torch.save(deeplab, 'deeplab.pth')
            s3 = boto3.client('s3')
            s3.upload_file('deeplab.pth', 'raster-vision-mcclain',
                           '{}/deeplab_first_and_last_{}epochs_{}channels_1.pth'.format(dataset_name, epochs, CHANNELS))

        print('\t TRAINING FIRST AND LAST LAYERS AGAIN')
        try:
            s3 = boto3.client('s3')
            s3.download_file(
                'raster-vision-mcclain', '{}/deeplab_first_and_last_{}epochs_{}channels_2.pth'.format(dataset_name, epochs, CHANNELS), 'deeplab.pth')
            deeplab = torch.load('deeplab.pth').to(device)
            print('\t\t SUCCESSFULLY RESTARTED')
        except:
            last_class = deeplab.classifier[4]
            last_class_aux = deeplab.aux_classifier[4]
            input_filters = deeplab.backbone.conv1
            for p in deeplab.parameters():
                p.requires_grad = False
            for p in last_class.parameters():
                p.requires_grad = True
            for p in last_class_aux.parameters():
                p.requires_grad = True
            for p in input_filters.parameters():
                p.requires_grad = True

            ps = []
            for n, p in deeplab.named_parameters():
                if p.requires_grad == True:
                    ps.append(p)
                else:
                    p.grad = None
            opt = torch.optim.SGD(ps, lr=0.001, momentum=0.9)
            train(deeplab, opt, obj, steps_per_epoch, epochs // 2, batch_size,
                  raster_ds, mask_ds, width, height, device)

            print('\t UPLOADING')
            torch.save(deeplab, 'deeplab.pth')
            s3 = boto3.client('s3')
            s3.upload_file('deeplab.pth', 'raster-vision-mcclain',
                           '{}/deeplab_first_and_last_{}epochs_{}channels_2.pth'.format(dataset_name, epochs, CHANNELS))

        print('\t TRAINING ALL LAYERS')
        try:
            s3 = boto3.client('s3')
            s3.download_file(
                'raster-vision-mcclain', '{}/deeplab_all_{}epochs_{}channels_1.pth'.format(dataset_name, epochs, CHANNELS), 'deeplab.pth')
            deeplab = torch.load('deeplab.pth').to(device)
            print('\t\t SUCCESSFULLY RESTARTED')
        except:
            for p in deeplab.parameters():
                p.requires_grad = True

            ps = []
            for n, p in deeplab.named_parameters():
                if p.requires_grad == True:
                    ps.append(p)
                else:
                    p.grad = None
            opt = torch.optim.SGD(ps, lr=0.01, momentum=0.9)

            train(deeplab, opt, obj, steps_per_epoch, epochs, batch_size,
                  raster_ds, mask_ds, width, height, device)

            print('\t UPLOADING')
            torch.save(deeplab, 'deeplab.pth')
            s3 = boto3.client('s3')
            s3.upload_file('deeplab.pth', 'raster-vision-mcclain',
                           '{}/deeplab_all_{}epochs_{}channels_1.pth'.format(dataset_name, epochs, CHANNELS))

        print('\t TRAINING ALL LAYERS AGAIN')
        for p in deeplab.parameters():
            p.requires_grad = True

        ps = []
        for n, p in deeplab.named_parameters():
            if p.requires_grad == True:
                ps.append(p)
            else:
                p.grad = None

        opt = torch.optim.SGD(ps, lr=0.001, momentum=0.9)

        train(deeplab, opt, obj, steps_per_epoch, epochs, batch_size,
              raster_ds, mask_ds, width, height, device)

        print('\t SAVING')
        torch.save(deeplab, 'deeplab.pth')
        s3 = boto3.client('s3')
        s3.upload_file('deeplab.pth', 'raster-vision-mcclain',
                       '{}/deeplab_all_{}epochs_{}channels.pth'.format(dataset_name, epochs, CHANNELS))

# ./download_run_upload.sh s3://raster-vision-mcclain/vegas/vegas_train.py vegas_train.py s3://raster-vision-mcclain/xxx 8 10
# ./download_run_upload.sh s3://raster-vision-mcclain/vegas/vegas_train.py vegas_train.py s3://raster-vision-mcclain/xxx 8 10 shanghai/data/MUL_AOI_4_Shanghai.tif shanghai/data/mask_AOI_4_Shanghai.tif shanghai
# ./download_run_upload.sh s3://raster-vision-mcclain/vegas/vegas_train.py vegas_train.py s3://raster-vision-mcclain/xxx 8 10 vegas_roads/data/MUL_AOI_2_Vegas.tif vegas_roads/data/mask_AOI_2_Vegas.tif vegas_roads
# ./download_run_upload.sh s3://raster-vision-mcclain/vegas/vegas_train.py vegas_train.py s3://raster-vision-mcclain/xxx 8 10 vegas/data/MUL_AOI_2_Vegas.tif vegas/data/mask_AOI_2_Vegas.tif vegas_norm 0.00430695255445 0.00663654054088 0.0088010822481 0.00657800217491 0.00631494305997 0.00622265943938 0.00825494408698 0.00528162575785 0.00421938530572 0.00686205490534 0.00948095772569 0.00736970404955 0.00725928605596 0.00681741366099 0.00908610352256 0.00581618206724
# ./download_run_upload.sh s3://raster-vision-mcclain/vegas/vegas_train.py vegas_train.py s3://raster-vision-mcclain/xxx 3 10 vegas/data/MUL_AOI_2_Vegas.tif vegas/data/mask_AOI_2_Vegas.tif vegas_norm 0.00631494305997 0.0088010822481 0.00663654054088 0.00725928605596 0.00948095772569 0.00686205490534
