"""
Mapbox vector map tile client.

Example usage:
```
from map_tile_client.models.mapbox_map import MapBoxVMap
mapbox_map = MapBoxVMap(37.33981343865369, -122.04560815516787)
mapbox_map.expand_border(4)
mapbox_map.get_map().show()
```
"""

from enum import Enum
import mapbox_vector_tile
import os
from PIL import Image, ImageDraw
import requests

from map_tile_client.api.mapbox_api import MapBoxAPI
import map_tile_client.cache
from map_tile_client.models._base_map import BaseMap
from map_tile_client.utils import color_utils, geometry_utils


MAPBOX_API = MapBoxAPI()


class MapBoxMapType(Enum):
    VMap = "VMap"


CACHE_DIR = os.path.join(list(map_tile_client.cache.__path__)[0], "mapbox")
TILE_CACHE_DIRS = {MapBoxMapType.VMap: os.path.join(CACHE_DIR, "vmap")}
for cache_dir in [CACHE_DIR] + list(TILE_CACHE_DIRS.values()):
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)


class MapBoxVMap(BaseMap):
    TILE_CACHE_DIR = TILE_CACHE_DIRS[MapBoxMapType.VMap]
    TILE_BASE_URL = "https://api.mapbox.com/v4/mapbox.mapbox-streets-v8"

    def __init__(
        self,
        latitude_deg,
        longitude_deg,
        zoom=20,
        load_from_cache=True,
        save_to_cache=True,
        multithread=True
    ):
        super().__init__(
            latitude_deg,
            longitude_deg,
            zoom=zoom,
            load_from_cache=load_from_cache,
            save_to_cache=save_to_cache,
            multithread=multithread
        )
        self.roads = {}

    def load_tiles(self, tile_keys):
        for x_delta, y_delta in tile_keys:
            self.load_tile(x_delta, y_delta)

    def get_vmap_url(self, x, y, zoom):
        road_url = (
            f"{MapBoxVMap.TILE_BASE_URL}/{zoom}/{x}/{y}.vector.pbf"
            f"?sku=101sHi7zTy5Qn&access_token={MAPBOX_API.get_access_token()}"
        )
        return road_url

    def get_roads(self, x_delta, y_delta):
        if (x_delta, y_delta) in self.roads:
            return self.roads[(x_delta, y_delta)]
        x, y = x_delta + self.origin_tile_x, y_delta + self.origin_tile_y
        vtile = mapbox_vector_tile.decode(requests.get(self.get_vmap_url(x, y, self.zoom)).content)

        self.roads[(x_delta, y_delta)] = {}
        if "road" in vtile:
            for feature in vtile["road"]["features"]:
                if feature["properties"]["class"] != "street" and feature["properties"]["type"] != "secondary":
                    continue
                if feature["geometry"]["type"] == "LineString":
                    coordinates = [feature["geometry"]["coordinates"]]
                elif feature["geometry"]["type"] == "MultiLineString":
                    coordinates = feature["geometry"]["coordinates"]
                else:
                    continue
                for i in range(len(coordinates)):
                    # By default, the road is returned in a tile of size 4096x4096
                    # Scale down to the conventional 256x256
                    coordinates[i] = [
                        (
                            256.0 * (float(p[0]) / 4096.0),
                            256.0 * (1 - (float(p[1]) / 4096.0)),
                        )
                        for p in coordinates[i]
                    ]
                street_name = (
                    feature["properties"]["name"] if "name" in feature["properties"] else feature["properties"]["type"]
                )
                if street_name not in self.roads[(x_delta, y_delta)]:
                    self.roads[(x_delta, y_delta)][street_name] = []
                self.roads[(x_delta, y_delta)][street_name] = self.roads[(x_delta, y_delta)][street_name] + coordinates
        return self.roads[(x_delta, y_delta)]

    def load_tile(self, x_delta, y_delta):
        if (x_delta, y_delta) in self.tiles:
            return
        if self.multithread:
            self.num_tiles_loading += 1
            self.load_map_lock.acquire()
            self.tile_load_queue.put((x_delta, y_delta))
            self.load_map_lock.release()
        else:
            tile_x, tile_y = self.origin_tile_x + x_delta, self.origin_tile_y + y_delta
            cache_path = self.get_tile_cache_path(tile_x, tile_y)
            if self.load_from_cache and os.path.exists(cache_path):
                self.tiles[(x_delta, y_delta)] = Image.open(cache_path)
            else:
                self.tiles[(x_delta, y_delta)] = Image.new("RGB", (256, 256))
                tile_draw = ImageDraw.Draw(self.tiles[(x_delta, y_delta)])
                # Uncomment this line to add a white border around each tile
                # image_bounds = geometry_utils.get_image_bounds(256, 256)
                # for image_edge_px in zip(image_bounds[:-1], image_bounds[1:]):
                #     tile_draw.line([tuple(px) for px in image_edge_px], fill=(255, 255, 255), width=2)
                for street_name, road_edges_px in self.get_roads(x_delta, y_delta).items():
                    for road_edge_px in road_edges_px:
                        road_uv_px = geometry_utils.bound_line(road_edge_px, 256.0, 256.0)
                        if len(road_uv_px) == 0:
                            continue
                        road_uv_px = geometry_utils.snap_to_image_boundary(
                            road_uv_px, 256.0, 256.0, tolerance=1.0, inside_and_out=True
                        )
                        print(road_uv_px)
                        tile_draw.line(
                            [(int(x[0]), int(x[1])) for x in road_uv_px],
                            fill=color_utils.get_color(street_name),
                            width=8,
                        )
                if self.save_to_cache:
                    self.tiles[(x_delta, y_delta)].save(cache_path)
