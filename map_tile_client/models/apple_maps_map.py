"""
Apple Maps map tile client.

Example usage:
```
from map_tile_client.models.apple_maps_map import AppleMapsSatelliteMap, AppleMapsStandardMap
apple_maps_map = AppleMapsStandardMap(37.33981343865369, -122.04560815516787)
apple_maps_map.expand_border(4)
apple_maps_map.get_map().show()
```
"""

import abc
from enum import Enum
import os

from map_tile_client.api.apple_maps_api import AppleMapsAPI
import map_tile_client.cache
from map_tile_client.models._base_map import BaseMap


class AppleMapsMapType(Enum):
    Satellite = "Satellite"
    Standard = "Standard"


APPLE_STANDARD_MAPS_BUILDINGS_COLOR = np.array([233, 233, 226])
APPLE_MAPS_API = AppleMapsAPI()
CACHE_DIR = os.path.join(list(map_tile_client.cache.__path__)[0], "apple_maps")
TILE_CACHE_DIRS = {
    AppleMapsMapType.Satellite: os.path.join(CACHE_DIR, "satellite_tiles"),
    AppleMapsMapType.Standard: os.path.join(CACHE_DIR, "standard_tiles")
}
TILE_BASE_URLS = {
    AppleMapsMapType.Satellite: "https://sat-cdn2.apple-mapkit.com/tile?style=7&size=1&scale=1&v=9262",
    AppleMapsMapType.Standard: "https://cdn2.apple-mapkit.com/ti/tile?style=0&size=1&scale=1"
}
for cache_dir in [CACHE_DIR] + list(TILE_CACHE_DIRS.values()):
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)


class AppleMapsBaseMap(BaseMap):

    def __init__(self, map_type, latitude_deg, longitude_deg, zoom=20,
                 load_from_cache=True, save_to_cache=True):
        AppleMapsBaseMap.TILE_BASE_URL = TILE_BASE_URLS[map_type]
        AppleMapsBaseMap.TILE_CACHE_DIR = TILE_CACHE_DIRS[map_type]
        super().__init__(latitude_deg, longitude_deg, zoom=zoom,
                         load_from_cache=load_from_cache, save_to_cache=save_to_cache)

    @property
    @abc.abstractmethod
    def TILE_CACHE_DIR(self):
        return NotImplementedError

    @property
    @abc.abstractmethod
    def TILE_BASE_URL(self):
        return NotImplementedError

    def get_tile_url(self, x, y):
        print(f"{x}, {y}, {self.zoom}")
        return f"{AppleMapsBaseMap.TILE_BASE_URL}&x={x}&y={y}&z={self.zoom}" \
               f"&accessKey={APPLE_MAPS_API.get_access_token()}"


class AppleMapsSatelliteMap(AppleMapsBaseMap):
    def __init__(self, latitude_deg, longitude_deg, zoom=20,
                 load_from_cache=True, save_to_cache=True):
        super().__init__(AppleMapsMapType.Satellite, latitude_deg, longitude_deg, zoom=zoom,
                         load_from_cache=load_from_cache, save_to_cache=save_to_cache)


class AppleMapsStandardMap(AppleMapsBaseMap):
    def __init__(self, latitude_deg, longitude_deg, zoom=20,
                 load_from_cache=True, save_to_cache=True):
        super().__init__(AppleMapsMapType.Standard, latitude_deg, longitude_deg, zoom=zoom,
                         load_from_cache=load_from_cache, save_to_cache=save_to_cache)

    def get_mono_map(self):
        apple_maps_mono_map = cv2.inRange(
            np.array(self.get_map()), APPLE_STANDARD_MAPS_BUILDINGS_COLOR - np.array([1, 1, 1]),
            APPLE_STANDARD_MAPS_BUILDINGS_COLOR + np.array([1, 1, 1])
        )
        apple_maps_mono_map = 255 - apple_maps_mono_map  # Apple maps need to invert
        return Image.fromarray(apple_maps_mono_map)
