import time

import numpy as np
from PIL import Image

import boto3
import torch
import torchvision

train_scenes = ['2_10', '2_11', '3_10', '3_11', '4_10', '4_11', '5_10', '5_11',
                '5_12', '6_10', '6_11', '6_12', '6_7', '6_8', '6_9', '7_10', '7_11', '7_12', '7_7', '7_8']
val_scenes = ['2_12', '3_12', '4_12', '7_9']
scenes = train_scenes + val_scenes
scenes = ['2_10']  # XXX

rgb_data = []
elevation_data = []
label_data = []


def download_data():
    s3 = boto3.client('s3')
    for scene in scenes:
        s3.download_file('raster-vision-raw-data',
                         'isprs-potsdam/5_Labels_for_participants/top_potsdam_{}_label.tif'
                         .format(scene),
                         '/tmp/labels_{}.tif'.format(scene))
        s3.download_file('raster-vision-raw-data',
                         'isprs-potsdam/1_DSM_normalisation/dsm_potsdam_0{}_normalized_lastools.jpg'
                         .format(scene),
                         '/tmp/elevation_{}.jpg'.format(scene))
        s3.download_file('raster-vision-mcclain',
                         'potsdam/top_potsdam_{}_RGB.tif'.format(scene),
                         '/tmp/rgb_{}.tif'.format(scene))
    del s3


def load_data():
    for scene in scenes:
        rgb_data.append(Image.open('/tmp/rgb_{}.tif'.format(scene)))
        elevation_data.append(Image.open(
            '/tmp/elevation_{}.jpg'.format(scene)))
        label_data.append(Image.open('/tmp/labels_{}.tif'.format(scene)))


def transmute_to_classes(window):
    # 2 = tree
    # 3 = ground
    # 4 = clutter
    # 1 = building
    # 5 = car
    # 0 = everything else
    fours = 4*(window[:, :, 0]/0xff).astype(np.long)
    twos = 2*(window[:, :, 1]/0xff).astype(np.long)
    ones = 1*(window[:, :, 2]/0xff).astype(np.long)
    retval = fours + twos + ones
    retval = retval * (retval < 6)  # White is seven, turn it to zero
    return retval


def random_potsdam_training_window():
    size = 224

    x = np.random.randint(0, 6000 - size)
    y = np.random.randint(0, 6000 - size)
    z = np.random.randint(0, len(scenes))

    box = (x, y, x + size, y + size)
    rgb_window = rgb_data[z].crop(box)
    elevation_window = elevation_data[z].crop(box)
    labels_window = label_data[z].crop(box)
    labels_window = transmute_to_classes(np.array(labels_window))

    return (rgb_window, elevation_window, labels_window)


def random_potsdam_training_batch():
    batch_size = 64

    rgbs = []
    elvs = []
    labs = []

    for i in range(batch_size):
        rgb, elv, lab = random_potsdam_training_window()

        rgbs.append(transforms3(rgb))
        elvs.append(transforms1(elv))
        labs.append(torch.unsqueeze(torch.from_numpy(lab), 0))

    rgbs = torch.stack(rgbs).to(device)
    elvs = torch.stack(elvs).to(device)
    labs = torch.cat(labs, dim=0).to(device)

    return (rgbs, elvs, labs)


if __name__ == "__main__":

    # RNG
    np.random.seed(seed=33)

    # GPU
    device = torch.device("cuda")

    # Network
    print('Network')
    deeplab_resnet101 = torchvision.models.segmentation.deeplabv3_resnet101(
        pretrained=True)

    # Transforms
    normalize3 = torchvision.transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    normalize1 = torchvision.transforms.Normalize(mean=[0.485], std=[0.229])
    transforms3 = torchvision.transforms.Compose(
        [torchvision.transforms.ToTensor(), normalize3])
    transforms1 = torchvision.transforms.Compose(
        [torchvision.transforms.ToTensor(), normalize1])

    # Download
    print('Download')
    if True:
        download_data()

    # Load Data
    print('Load Data')
    load_data()

    # Reshape Network for 6 Classes
    last_class = deeplab_resnet101.classifier[4] = torch.nn.Conv2d(
        256, 6, kernel_size=(1, 1), stride=(1, 1))
    last_class_aux = deeplab_resnet101.aux_classifier[4] = torch.nn.Conv2d(
        256, 6, kernel_size=(1, 1), stride=(1, 1))

    # Feature Extraction Only
    for p in deeplab_resnet101.parameters():
        p.requires_grad = False
    for p in last_class.parameters():
        p.requires_grad = True
    for p in last_class_aux.parameters():
        p.requires_grad = True

    # Send network to device
    print('Network to Device')
    deeplab_resnet101 = deeplab_resnet101.to(device)

    # Optimizer
    ps = []
    for n, p in deeplab_resnet101.named_parameters():
        if p.requires_grad == True:
            ps.append(p)
    opt = torch.optim.SGD(ps, lr=0.01, momentum=0.9)

    # Objective Function
    obj = torch.nn.CrossEntropyLoss().to(device)

    # Train
    print('Train')
    steps_per_epoch_per_image = int((6000 * 6000) / (224 * 224 * 64))
    epochs = 10
    deeplab_resnet101.train()
    for i in range(epochs):
        for j in range(steps_per_epoch_per_image * len(scenes)):
            batch_tensor = random_potsdam_training_batch()
            opt.zero_grad()
            pred = deeplab_resnet101(batch_tensor[0])
            loss = obj(
                pred.get('out'), batch_tensor[2]) + 0.4*obj(pred.get('aux'), batch_tensor[2])
            loss.backward()
            opt.step()
        print('epoch={} time={} loss={}'.format(i, time.time(), loss.item()))

    # Save
    print('Save')
    torch.save(deeplab_resnet101, 'deeplab_resnet101.pth')
