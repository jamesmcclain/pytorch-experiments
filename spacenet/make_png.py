import sys

import numpy as np
import rasterio as rio
from PIL import Image


MEANS = [282.25613565591533, 434.92568434644414, 576.778925128917, 431.0893725326917,
         413.849793435062, 407.8019863596957, 540.9877607404478, 346.13134404092824]
STDS = [93.98737081435225, 204.11210503974044, 321.96208260604055, 274.7927091658248,
        284.49701298475304, 241.7486663735505, 325.7805731808835, 208.77036985582086]
MAGIC_NUMBER = 2**16 - 1
OTHER_MAGIC_NUMBER = 2**8 - 1
WINDOW_SIZE = 1024
CHANNELS = 8

np.random.seed(seed=33)


if __name__ == "__main__":

    with rio.open(sys.argv[1]) as raster_ds, rio.open(sys.argv[2]) as mask_ds:
        if CHANNELS == 3 or True:
            bands = [2, 3, 5]
        else:
            bands = raster_ds.indexes[0:CHANNELS]

        # Nodata
        nodata = raster_ds.read(1) == MAGIC_NUMBER
        not_nodata = (nodata == 0)

        # Labels
        labels = mask_ds.read(1) == OTHER_MAGIC_NUMBER
        labels = (labels*not_nodata) + (2*nodata)  # class 2 ignored
        labels = np.uint8(labels*127)

        # Rasters
        data = []
        for band in bands:
            a = raster_ds.read(band)
            a = np.array((a - MEANS[band-1]) / STDS[band-1], dtype=np.float32)
            a = a * not_nodata
            data.append(a)
        data = np.stack(data, axis=2)
        data = data*0.125 + 0.5
        data = np.uint8(data*255)

        # Save
        Image.fromarray(data).save('/tmp/rgb.png')
        Image.fromarray(labels).save('/tmp/labels.png')
        both = np.stack([data[:,:,0], data[:,:,1], labels], axis=2)
        Image.fromarray(both).save('/tmp/both.png')
