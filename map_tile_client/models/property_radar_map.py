"""
Class to load/manage/process map data from PropertyRadar.
"""
import cv2
from enum import Enum
import numpy as np
import os
from PIL import Image

import map_tile_client.cache
from map_tile_client.models._base_map import BaseMap


class PropertyRadarMapType(Enum):
    Street = "Street"
    Parcel = "Parcel"


CACHE_DIR = os.path.join(list(map_tile_client.cache.__path__)[0], "property_radar")
TILE_CACHE_DIRS = {
    PropertyRadarMapType.Street: os.path.join(CACHE_DIR, "street_tile"),
    PropertyRadarMapType.Parcel: os.path.join(CACHE_DIR, "parcel_tiles"),
}
for cache_dir in [CACHE_DIR] + list(TILE_CACHE_DIRS.values()):
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)


class PropertyRadarStreetMap(BaseMap):
    TILE_CACHE_DIR = TILE_CACHE_DIRS[PropertyRadarMapType.Street]
    TILE_BASE_URL = "https://prn-cdn-c.propertyradar.com/t/propertyType"

    def __init__(
        self,
        latitude_deg,
        longitude_deg,
        zoom=20,
        load_from_cache=True,
        save_to_cache=True,
    ):
        super().__init__(
            latitude_deg,
            longitude_deg,
            zoom=zoom,
            load_from_cache=load_from_cache,
            save_to_cache=save_to_cache,
        )

    def get_tile_url(self, x, y):
        return f"{PropertyRadarStreetMap.TILE_BASE_URL}/{self.zoom}/{x}/{y}.png"


class PropertyRadarParcelMap(BaseMap):
    TILE_CACHE_DIR = TILE_CACHE_DIRS[PropertyRadarMapType.Parcel]
    TILE_BASE_URL = "https://prn-cdn-b.propertyradar.com/t/parcelBounds"

    def __init__(
        self,
        latitude_deg,
        longitude_deg,
        zoom=20,
        multithread=True,
        load_from_cache=True,
        save_to_cache=True,
    ):
        super().__init__(
            latitude_deg,
            longitude_deg,
            zoom=zoom,
            load_from_cache=load_from_cache,
            save_to_cache=save_to_cache,
        )
        self.multithread = multithread
        self.enclosing_tile_dims = [0, 0, 0, 0]  # Left, Top, Right, Bottom
        self._init_parcel_tiles()  # Set self.num_tiles_expand

    def get_tile_url(self, x, y):
        return f"{PropertyRadarParcelMap.TILE_BASE_URL}/{self.zoom}/{x}/{y}.png"

    def get_mono_map(self):
        mono_map = np.array(self.get_map())
        mono_map = cv2.inRange(mono_map, np.array([0, 0, 0]), np.array([0, 0, 0]))
        mono_map_mask = np.zeros(tuple((np.array(mono_map.shape) + 2)), np.uint8)
        property_center = (
            np.array(self.origin_uv_px)
            + np.array([255 * self.enclosing_tile_dims[0], 255 * self.enclosing_tile_dims[1]])
        ).tolist()
        cv2.floodFill(mono_map, mono_map_mask, [int(p) for p in property_center], 128)
        mono_map = cv2.inRange(mono_map, 128, 128)
        return Image.fromarray(mono_map)

    def get_enclosing_tiles(self):
        enclosing_tiles = []
        for x_delta in range(-self.enclosing_tile_dims[0], self.enclosing_tile_dims[2] + 1):
            for y_delta in range(-self.enclosing_tile_dims[1], self.enclosing_tile_dims[3] + 1):
                enclosing_tiles.append((x_delta, y_delta))
        return enclosing_tiles

    def _init_parcel_tiles(self):
        complete = [False, False, False, False]  # Left, Top, Right, Bottom

        def load_parcel_map_border():
            for x_delta in range(-self.enclosing_tile_dims[0], self.enclosing_tile_dims[2] + 1):
                self.load_tile(x_delta, -self.enclosing_tile_dims[1], multithread=self.multithread)
                self.load_tile(x_delta, self.enclosing_tile_dims[3], multithread=self.multithread)
            for y_delta in range(-self.enclosing_tile_dims[1] + 1, self.enclosing_tile_dims[3]):
                self.load_tile(-self.enclosing_tile_dims[0], y_delta, multithread=self.multithread)
                self.load_tile(self.enclosing_tile_dims[2], y_delta, multithread=self.multithread)

        def update_complete_vars():
            nonlocal complete
            mono_map = np.array(self.get_mono_map())
            sides = [
                mono_map[:, 0],  # Left
                mono_map[0, :],  # Top
                mono_map[:, -1],  # Right
                mono_map[-1, :],  # Bottom
            ]
            for i in range(4):
                if not complete[i] and all(sides[i] == 0):
                    complete[i] = True

        self.load_tile(0, 0, multithread=False)
        update_complete_vars()
        while not all(complete):
            for i in range(4):
                if not complete[i]:
                    self.enclosing_tile_dims[i] += 1
            load_parcel_map_border()
            update_complete_vars()
