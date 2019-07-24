import time

import numpy as np
from PIL import Image

import boto3
import torch
import torchvision

val_scenes = ['2_12', '3_12', '4_12']
scenes = val_scenes

rgb_data = []
elevation_data = []
label_data = []

normalize3 = torchvision.transforms.Normalize(
    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
normalize1 = torchvision.transforms.Normalize(mean=[0.485], std=[0.229])
transforms3 = torchvision.transforms.Compose(
    [torchvision.transforms.ToTensor(), normalize3])
transforms1 = torchvision.transforms.Compose(
    [torchvision.transforms.ToTensor(), normalize1])


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


def download_model():
    s3 = boto3.client('s3')
    s3.download_file('raster-vision-mcclain',
                     'potsdam/deeplab_resnet101_rgb.pth', '/tmp/deeplab_resnet101_rgb.pth')
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
    retval = (4*(window[:, :, 0]/0xff).astype(np.long) + 2*(window[:, :, 1] /
                                                            0xff).astype(np.long) + 1*(window[:, :, 2]/0xff).astype(np.long))
    cars = (retval == 6)
    not_cars = (retval != 6)
    retval = (retval * not_cars) + 5*cars
    retval = retval * (retval < 6)
    return retval


def potsdam_eval_window(x, y, rgb_data, elevation_data, label_data):
    size = 1000
    x = int(x * size)
    y = int(y * size)
    box = (x, y, x + size, y + size)
    rgb_window = rgb_data.crop(box)
    elevation_window = elevation_data.crop(box)
    labels_window = np.array(label_data.crop(box))
    labels_window = transmute_to_classes(labels_window)
    return (rgb_window, elevation_window, labels_window)


def potsdam_eval_batch(x, y, rgb_ar, elevation_ar, labels_ar):
    batch_size = 1

    rgbs = []
    elvs = []
    labs = []

    for i in range(batch_size):
        rgb, elv, lab = potsdam_eval_window(
            x, y, rgb_ar, elevation_ar, labels_ar)

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
    print('Model')
    if True:
        download_model()
    deeplab_resnet101 = torch.load(
        '/tmp/deeplab_resnet101_rgb.pth').to(device)
    deeplab_resnet101.eval()

    # Download
    print('Download Data')
    if True:
        download_data()

    # Load Data
    print('Load Data')
    load_data()

    # Compute True Positives, False Positives, False Negatives
    tps = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    fps = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    fns = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    for i in range(len(rgb_data)):
        for x in range(6):
            for y in range(6):
                batch_tensor = potsdam_eval_batch(
                    x, y, rgb_data[i], elevation_data[i], label_data[i])
                out = deeplab_resnet101(batch_tensor[0])
                out = out['out'].data.cpu().numpy()
                index = 0
                predicted_segments = np.apply_along_axis(
                    np.argmax, 0, out[index])
                groundtruth_segments = batch_tensor[2].data.cpu().numpy()[
                    index]
                for c in range(6):
                    tps[c] += ((predicted_segments == c) *
                               (groundtruth_segments == c)).sum()
                    fps[c] += ((predicted_segments == c) *
                               (groundtruth_segments != c)).sum()
                    fns[c] += ((predicted_segments != c) *
                               (groundtruth_segments == c)).sum()
    with open('output.txt', 'a') as f:
        f.write('True Positives:  {}\n'.format(tps))
        f.write('False Positives: {}\n'.format(fps))
        f.write('False Negatives: {}\n'.format(fns))
    print('True Positives:  {}'.format(tps))
    print('False Positives: {}'.format(fps))
    print('False Negatives: {}'.format(fns))

    # Recalls, Precisions
    recalls = []
    precisions = []
    for c in range(6):
        recall = tps[c] / (tps[c] + fns[c])
        recalls.append(recall)
        precision = tps[c] / (tps[c] + fps[c])
        precisions.append(precision)
    with open('output.txt', 'a') as f:
        f.write('Recalls:    {}\n'.format(recalls))
        f.write('Precisions: {}\n'.format(precisions))
    print('Recalls:    {}'.format(recalls))
    print('Precisions: {}'.format(precisions))

    # f1 Scores
    names = [
        'other         ',
        'building      ',
        'tree          ',
        'low vegetation',
        'clutter       ',
        'car           '
    ]
    f1s = []
    for c in range(6):
        f1 = 2 * (precisions[c] * recalls[c]) / (precisions[c] + recalls[c])
        f1s.append(f1)
        with open('output.txt', 'a') as f:
            f.write('{} {}\n'.format(names[c], f1))
        print('{} {}'.format(names[c], f1))

    # Overall f1 Scores
    precision = np.array(tps).sum() / (np.array(tps).sum() + np.array(fps).sum())
    recall = np.array(tps).sum() / (np.array(tps).sum() + np.array(fns).sum())
    f1 = 2 * (precision * recall) / (precision + recall)
    with open('output.txt', 'a') as f:
        f.write('Overall Precision: {}\n'.format(precision))
        f.write('Overall Recall:    {}\n'.format(recall))
        f.write('Overall f1:        {}\n'.format(f1))
    print('Overall Precision: {}'.format(precision))
    print('Overall Recall:    {}'.format(recall))
    print('Overall f1:        {}'.format(f1))

# ./download_run_upload.sh s3://bucket/potsdam/eval.py output.txt s3://bucket/potsdam/potsdam-eval.txt
