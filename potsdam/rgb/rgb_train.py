import time

import numpy as np
from PIL import Image

import boto3
import torch
import torchvision

train_scenes = ['2_10', '2_11', '3_10', '3_11', '4_10', '4_11', '5_10', '5_11',
                '5_12', '6_10', '6_11', '6_12', '7_10', '7_11', '7_12']
scenes = train_scenes

rgb_data = []
elevation_data = []
label_data = []

image_size = 224
batch_size = 64

normalize3 = torchvision.transforms.Normalize(
    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
normalize1 = torchvision.transforms.Normalize(mean=[0.485], std=[0.229])
transforms3 = torchvision.transforms.Compose(
    [torchvision.transforms.ToTensor(), normalize3])
transforms1 = torchvision.transforms.Compose(
    [torchvision.transforms.ToTensor(), normalize1])


def train(model, opt, obj, epochs):
    steps_per_epoch_per_image = int(
        (6000 * 6000) / (image_size * image_size * batch_size))
    steps_per_epoch = steps_per_epoch_per_image * len(scenes)
    model.train()
    current_time = time.time()
    for i in range(epochs):
        avg_loss = 0.0
        for j in range(steps_per_epoch):
            batch_tensor = random_potsdam_training_batch()
            opt.zero_grad()
            pred = model(batch_tensor[0])
            loss = obj(
                pred.get('out'), batch_tensor[2]) + 0.4*obj(pred.get('aux'), batch_tensor[2])
            loss.backward()
            opt.step()
            avg_loss = avg_loss + loss.item()
        avg_loss = avg_loss / steps_per_epoch
        last_time = current_time
        current_time = time.time()
        print('epoch={} time={} avg_loss={}'.format(
            i, current_time - last_time, avg_loss))


def download_data():
    s3 = boto3.client('s3')
    for scene in scenes:
        print('scene={}'.format(scene))
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
    cars = (retval == 6)
    not_cars = (retval != 6)
    retval = (retval * not_cars) + 5*cars # Change cars from class 6 to class 5
    retval = retval * (retval < 6)  # White is seven, turn it to zero
    return retval


def random_potsdam_training_window():
    x = np.random.randint(0, 6000 - image_size)
    y = np.random.randint(0, 6000 - image_size)
    z = np.random.randint(0, len(scenes))

    box = (x, y, x + image_size, y + image_size)
    rgb_window = rgb_data[z].crop(box)
    elevation_window = elevation_data[z].crop(box)
    labels_window = label_data[z].crop(box)
    labels_window = transmute_to_classes(np.array(labels_window))

    return (rgb_window, elevation_window, labels_window)


def random_potsdam_training_batch():
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

    ####### TRAIN LAST LAYER #######

    # Last Layer Only
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

    print('Training Last Layer')
    train(deeplab_resnet101, opt, obj, 10)

    ####### TRAIN ALL LAYERS #######

    batch_size = 16

    # All Layers
    for p in deeplab_resnet101.parameters():
        p.requires_grad = True

    # Optimizer
    ps = []
    for n, p in deeplab_resnet101.named_parameters():
        if p.requires_grad == True:
            ps.append(p)
    opt = torch.optim.SGD(ps, lr=0.01, momentum=0.9)

    print('Training All Layers')
    train(deeplab_resnet101, opt, obj, 20)

    print('Training All Layers Again')
    opt = torch.optim.SGD(ps, lr=0.001, momentum=0.9)
    train(deeplab_resnet101, opt, obj, 20)

    ####### SAVE #######

    print('Save')
    torch.save(deeplab_resnet101, 'deeplab_resnet101.pth')

# ./download_run_upload.sh s3://raster-vision-mcclain/potsdam/rge_train.py deeplab_resnet101.pth s3://raster-vision-mcclain/potsdam/deeplab_resnet101_rgb.pth
