import json
import sys

import shapely_geojson
from shapely.geometry import GeometryCollection, shape
from shapely.ops import cascaded_union

if __name__ == "__main__":
    with open(sys.argv[1]) as f:
        features = json.load(f)['features']

    polygons = [shape(feature["geometry"]).buffer(10/111111.0) for feature in features]
    polygon = cascaded_union(polygons)

    with open(sys.argv[2], 'w') as f:
        shapely_geojson.dump(polygon, f)
