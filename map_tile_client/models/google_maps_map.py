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
import cv2
from enum import Enum
import numpy as np
import os
from PIL import Image

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


GOOGLE_STANDARD_MAPS_ROOF_COLOR = np.array([241, 241, 241])
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

    # from map_tile_client.models.google_maps_map import GoogleMapsStandardMap
    # google_maps_map = GoogleMapsStandardMap(37.33981343865369, -122.04560815516787)
    # google_maps_map.expand_border(2)
    # google_maps_map.get_roofs_mono_map().show()
    def get_roofs_mono_map(self):
        mono_map = cv2.inRange(
            np.array(self.get_map()),
            GOOGLE_STANDARD_MAPS_ROOF_COLOR - np.array([1, 1, 1]),
            GOOGLE_STANDARD_MAPS_ROOF_COLOR + np.array([1, 1, 1]),
        )
        kernel = np.ones((3, 3), np.uint8)
        opening = cv2.morphologyEx(mono_map, cv2.MORPH_OPEN, kernel)
        closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(closing, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        smooth_image = np.zeros_like(mono_map)
        for contour in contours:
            epsilon = 0.01 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            cv2.drawContours(smooth_image, [approx], -1, (255), thickness=cv2.FILLED)
        return Image.fromarray(cv2.bitwise_not(smooth_image))


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
