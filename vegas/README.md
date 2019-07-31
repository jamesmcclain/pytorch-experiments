[geojson-merge](https://github.com/mapbox/geojson-merge)
[`gdal_merge.py -init 65535 -o MUL_AOI_2_Vegas.tif MUL/*`](https://gdal.org/programs/gdal_merge.html)
[`gdal_rasterize -burn 255 -ot Byte -a_srs '+proj=longlat +datum=WGS84 +no_defs ' -te -115.3075176 36.2616777 -115.1513226 36.1212777 -ts 14507 13040 geojson/buildings_AOI_2_Vegas_img1000.geojson mask.tif`](https://gdal.org/programs/gdal_rasterize.html)
https://spacenetchallenge.github.io/datasets/spacenetBuildings-V2summary.html
