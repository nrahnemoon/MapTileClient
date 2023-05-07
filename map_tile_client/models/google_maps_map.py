"""
Google Maps map tile client.

Example usage:
```
from map_tile_client.models.google_maps_map import GoogleMapsRoadMap, GoogleMapsStandardMap, \
  GoogleMapsHybridTerrainMap, GoogleMapsSatelliteMap, GoogleMapsTerrainMap, GoogleMapsHybridMap
google_maps_map = GoogleMapsSatelliteMap(37.33981343865369, -122.04560815516787)
google_maps_map.expand_border(4)
google_maps_map.get_map().show()
```
"""

import abc
from enum import Enum
import os

import map_tile_client.cache
from map_tile_client.models._base_map import BaseMap


class GoogleMapsMapType(Enum):
    Road = "h"
    Standard = "m"
    HybridTerrain = "p"
    Altered = "r"
    Satellite = "s"
    Terrain = "t"
    Hybrid = "y"


CACHE_DIR = os.path.join(list(map_tile_client.cache.__path__)[0], "google_maps")
TILE_CACHE_DIRS = {
    GoogleMapsMapType.Road: os.path.join(CACHE_DIR, "road_tiles"),
    GoogleMapsMapType.Standard: os.path.join(CACHE_DIR, "standard_tiles"),
    GoogleMapsMapType.HybridTerrain: os.path.join(CACHE_DIR, "hybrid_terrain_tiles"),
    GoogleMapsMapType.Satellite: os.path.join(CACHE_DIR, "satellite_tiles"),
    GoogleMapsMapType.Terrain: os.path.join(CACHE_DIR, "terrain_tiles"),
    GoogleMapsMapType.Hybrid: os.path.join(CACHE_DIR, "hybrid_tiles"),
}
TILE_BASE_URLS = {map_type: f"https://mt1.google.com/vt/lyrs={map_type.value}" for map_type in GoogleMapsMapType}
for cache_dir in [CACHE_DIR] + list(TILE_CACHE_DIRS.values()):
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)


class GoogleMapsBaseMap(BaseMap):
    def __init__(
        self,
        map_type,
        latitude_deg,
        longitude_deg,
        zoom=20,
        load_from_cache=True,
        save_to_cache=True,
    ):
        GoogleMapsBaseMap.TILE_BASE_URL = TILE_BASE_URLS[map_type]
        GoogleMapsBaseMap.TILE_CACHE_DIR = TILE_CACHE_DIRS[map_type]
        super().__init__(
            latitude_deg,
            longitude_deg,
            zoom=zoom,
            load_from_cache=load_from_cache,
            save_to_cache=save_to_cache,
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
        return f"{GoogleMapsBaseMap.TILE_BASE_URL}&x={x}&y={y}&z={self.zoom}"


class GoogleMapsRoadMap(GoogleMapsBaseMap):
    def __init__(
        self,
        latitude_deg,
        longitude_deg,
        zoom=20,
        load_from_cache=True,
        save_to_cache=True,
    ):
        super().__init__(
            GoogleMapsMapType.Road,
            latitude_deg,
            longitude_deg,
            zoom=zoom,
            load_from_cache=load_from_cache,
            save_to_cache=save_to_cache,
        )


class GoogleMapsStandardMap(GoogleMapsBaseMap):
    def __init__(
        self,
        latitude_deg,
        longitude_deg,
        zoom=20,
        load_from_cache=True,
        save_to_cache=True,
    ):
        super().__init__(
            GoogleMapsMapType.Standard,
            latitude_deg,
            longitude_deg,
            zoom=zoom,
            load_from_cache=load_from_cache,
            save_to_cache=save_to_cache,
        )


class GoogleMapsHybridTerrainMap(GoogleMapsBaseMap):
    def __init__(
        self,
        latitude_deg,
        longitude_deg,
        zoom=20,
        load_from_cache=True,
        save_to_cache=True,
    ):
        super().__init__(
            GoogleMapsMapType.HybridTerrain,
            latitude_deg,
            longitude_deg,
            zoom=zoom,
            load_from_cache=load_from_cache,
            save_to_cache=save_to_cache,
        )


class GoogleMapsSatelliteMap(GoogleMapsBaseMap):
    def __init__(
        self,
        latitude_deg,
        longitude_deg,
        zoom=20,
        load_from_cache=True,
        save_to_cache=True,
    ):
        super().__init__(
            GoogleMapsMapType.Satellite,
            latitude_deg,
            longitude_deg,
            zoom=zoom,
            load_from_cache=load_from_cache,
            save_to_cache=save_to_cache,
        )


class GoogleMapsTerrainMap(GoogleMapsBaseMap):
    def __init__(
        self,
        latitude_deg,
        longitude_deg,
        zoom=20,
        load_from_cache=True,
        save_to_cache=True,
    ):
        super().__init__(
            GoogleMapsMapType.Terrain,
            latitude_deg,
            longitude_deg,
            zoom=zoom,
            load_from_cache=load_from_cache,
            save_to_cache=save_to_cache,
        )


class GoogleMapsHybridMap(GoogleMapsBaseMap):
    def __init__(
        self,
        latitude_deg,
        longitude_deg,
        zoom=20,
        load_from_cache=True,
        save_to_cache=True,
    ):
        super().__init__(
            GoogleMapsMapType.Hybrid,
            latitude_deg,
            longitude_deg,
            zoom=zoom,
            load_from_cache=load_from_cache,
            save_to_cache=save_to_cache,
        )
