"""
Google Maps map tile client.

Example usage:
```
from map_tile_client.models.usgs_map import USGSStandardMap, USGSSatelliteMap
usgs_map = USGSSatelliteMap(37.33981343865369, -122.04560815516787, zoom=18)
usgs_map.expand_border(4)
usgs_map.get_map().show()
```
"""
import abc
import cv2
from enum import Enum
import numpy as np
import os
from PIL import Image

import map_tile_client.cache
from map_tile_client.models._base_map import BaseMap


class USGSMapType(Enum):
    Standard = "m"
    Satellite = "s"


CACHE_DIR = os.path.join(list(map_tile_client.cache.__path__)[0], "usgs")
TILE_CACHE_DIRS = {
    USGSMapType.Standard: os.path.join(CACHE_DIR, "standard_tiles"),
    USGSMapType.Satellite: os.path.join(CACHE_DIR, "satellite_tiles")
}
TILE_BASE_URLS = {
    USGSMapType.Standard: "https://services.arcgisonline.com/ArcGIS/rest/services/Reference/World_Boundaries_and_Places/MapServer/tile",
    USGSMapType.Satellite: "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile",
}
for cache_dir in [CACHE_DIR] + list(TILE_CACHE_DIRS.values()):
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)


class USGSBaseMap(BaseMap):
    def __init__(
        self,
        map_type,
        latitude_deg,
        longitude_deg,
        zoom=20,
        load_from_cache=True,
        save_to_cache=True,
        multithread=False
    ):
        USGSBaseMap.TILE_BASE_URL = TILE_BASE_URLS[map_type]
        USGSBaseMap.TILE_CACHE_DIR = TILE_CACHE_DIRS[map_type]
        super().__init__(
            latitude_deg,
            longitude_deg,
            zoom=zoom,
            load_from_cache=load_from_cache,
            save_to_cache=save_to_cache,
            multithread=multithread
        )

    @property
    @abc.abstractmethod
    def TILE_CACHE_DIR(self):
        return NotImplementedError

    @property
    @abc.abstractmethod
    def TILE_BASE_URL(self):
        return NotImplementedError

    def get_tile_url(self, x, y):
        return f"{USGSBaseMap.TILE_BASE_URL}/{self.zoom}/{y}/{x}"


class USGSStandardMap(USGSBaseMap):
    def __init__(
        self,
        latitude_deg,
        longitude_deg,
        zoom=20,
        load_from_cache=True,
        save_to_cache=True,
        multithread=False
    ):
        super().__init__(
            USGSMapType.Standard,
            latitude_deg,
            longitude_deg,
            zoom=zoom,
            load_from_cache=load_from_cache,
            save_to_cache=save_to_cache,
            multithread=multithread
        )


class USGSSatelliteMap(USGSBaseMap):
    def __init__(
        self,
        latitude_deg,
        longitude_deg,
        zoom=20,
        load_from_cache=True,
        save_to_cache=True,
        multithread=False
    ):
        super().__init__(
            USGSMapType.Satellite,
            latitude_deg,
            longitude_deg,
            zoom=zoom,
            load_from_cache=load_from_cache,
            save_to_cache=save_to_cache,
            multithread=multithread
        )
