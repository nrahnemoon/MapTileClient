"""
Abstract class to provide generic tile load and processing functionality.
"""
from __future__ import annotations
import abc
from io import BytesIO
import numpy as np
import os
from PIL import Image, ImageDraw, ImageOps
from queue import Queue
import requests
import threading
import time

from map_tile_client.utils import color_utils, geometry_utils, tilemap_utils


NUM_TILE_LOAD_THREADS = 8


class BaseMap:
    def __init__(
        self,
        latitude_deg,
        longitude_deg,
        zoom=20,
        load_from_cache=True,
        save_to_cache=True,
        multithread=False
    ):
        self.latitude_deg, self.longitude_deg, self.zoom = (
            latitude_deg,
            longitude_deg,
            zoom,
        )
        self.origin_tile_x, self.origin_tile_y, _ = tilemap_utils.get_tile(latitude_deg, longitude_deg, zoom)
        self.origin_uv_px = tilemap_utils.get_px_on_tile(latitude_deg, longitude_deg, zoom)
        self.load_from_cache, self.save_to_cache = load_from_cache, save_to_cache
        self.num_tiles_loading = 0
        self.tiles = {}
        self.load_map_lock = threading.Lock()
        self.map = Image.new("RGB", (0, 0))
        self.map_tiles = []  # (x_delta, y_delta) of tiles loaded into map
        self.multithread = multithread
        if multithread:
            self.tile_load_queue = Queue()
            self._start_tile_load_threads()

    @property
    @abc.abstractmethod
    def TILE_CACHE_DIR(self):
        return NotImplementedError

    @property
    @abc.abstractmethod
    def TILE_BASE_URL(self):
        return NotImplementedError

    @classmethod
    def from_map(cls, other_map: BaseMap, load_tiles=True):
        curr_map = cls(
            other_map.latitude_deg,
            other_map.longitude_deg,
            other_map.zoom,
            other_map.load_from_cache,
            other_map.save_to_cache,
        )
        if load_tiles:
            curr_map.load_tiles(other_map.get_tile_keys())
        return curr_map

    def get_tile_cache_path(self, x, y):
        return os.path.join(self.TILE_CACHE_DIR, f"{self.zoom}_{x}_{y}.png")

    @abc.abstractmethod
    def get_tile_url(self, x, y):
        raise NotImplementedError  # Note: zoom stored in self.zoom

    @abc.abstractmethod
    def get_mono_map(self):
        raise NotImplementedError  # Note: zoom stored in self.zoom

    def _start_tile_load_threads(self):
        for _ in range(NUM_TILE_LOAD_THREADS):
            tile_load_thread = threading.Thread(target=self._load_tile_from_queue, args=(self.tile_load_queue,))
            tile_load_thread.daemon = True
            tile_load_thread.start()

    def _load_tile_from_queue(self, load_tile_queue):
        while True:
            (x_delta, y_delta) = load_tile_queue.get()
            self.load_tile(x_delta, y_delta)
            self.num_tiles_loading -= 1

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
            loaded_from_cache = False
            if self.load_from_cache and os.path.exists(cache_path):
                if os.path.getsize(cache_path) == 0:
                    os.remove(cache_path)
                else:
                    self.tiles[(x_delta, y_delta)] = Image.open(cache_path)
                    loaded_from_cache = True
                    try:
                        self.tiles[(x_delta, y_delta)].load()
                    except OSError as e:
                        os.remove(cache_path)
                        loaded_from_cache = False
            if not loaded_from_cache:
                tile_url = self.get_tile_url(tile_x, tile_y)
                self.tiles[(x_delta, y_delta)] = Image.open(BytesIO(requests.get(tile_url).content))
                if self.save_to_cache:
                    self.tiles[(x_delta, y_delta)].save(cache_path)

    def load_tiles(self, tile_keys):
        for x_delta, y_delta in tile_keys:
            self.load_tile(x_delta, y_delta)

    def expand_border(self, num_tiles_expand):
        num_tiles_expand = int(num_tiles_expand)
        num_tiles_width_px, num_tiles_height_px = BaseMap.get_dimensions_px(self.tiles.keys())
        tiles_top_left_x_delta, tiles_top_left_y_delta = BaseMap.get_top_left_delta(self.tiles.keys())
        tiles_top_left_x_delta += num_tiles_expand
        tiles_top_left_y_delta += num_tiles_expand
        num_tiles_width = int(num_tiles_width_px / 256) + (2 * num_tiles_expand)
        num_tiles_height = int(num_tiles_height_px / 256) + (2 * num_tiles_expand)
        tiles = []
        for x_delta in range(-tiles_top_left_x_delta, num_tiles_width - tiles_top_left_x_delta):
            for y_delta in range(-tiles_top_left_y_delta, num_tiles_height - tiles_top_left_y_delta):
                tiles.append((x_delta, y_delta))
        self.load_tiles(tiles)

    @staticmethod
    def get_dimensions_px(tile_keys):
        deltas = np.array(list(tile_keys)).T.tolist()
        if len(deltas) == 0:
            return 0, 0
        width_px = int(256.0 * (max(deltas[0]) - min(deltas[0]) + 1))
        height_px = int(256.0 * (max(deltas[1]) - min(deltas[1]) + 1))
        return width_px, height_px

    @staticmethod
    def get_top_left_delta(tile_keys):
        deltas = np.array(list(tile_keys)).T.tolist()
        if len(deltas) == 0:
            return 0, 0
        return int(-min(deltas[0])), int(-min(deltas[1]))

    def get_map(self):
        self.load_map_lock.acquire()
        self.wait_tiles_loaded()

        # If no new tiles loaded, return current map
        if len(self.map_tiles) == len(self.tiles):
            self.load_map_lock.release()
            return self.map

        # Expand the self.map image so it can fit all the new self.tiles
        tiles_top_left_x_delta, tiles_top_left_y_delta = BaseMap.get_top_left_delta(self.tiles.keys())
        map_top_left_x_delta, map_top_left_y_delta = BaseMap.get_top_left_delta(self.map_tiles)
        if map_top_left_x_delta != tiles_top_left_x_delta:  # Expand the map left
            left_expand_px = (tiles_top_left_x_delta - map_top_left_x_delta) * 256
            self.map = ImageOps.expand(self.map, border=(left_expand_px, 0, 0, 0), fill=(0, 0, 0))
        if map_top_left_y_delta != tiles_top_left_y_delta:  # Expand the map top
            top_expand_px = (tiles_top_left_y_delta - map_top_left_y_delta) * 256
            self.map = ImageOps.expand(self.map, border=(0, top_expand_px, 0, 0), fill=(0, 0, 0))
        tiles_width_px, tiles_height_px = BaseMap.get_dimensions_px(self.tiles.keys())
        map_width_px, map_height_px = self.map.size
        if map_width_px != tiles_width_px:  # Expand the map right
            right_expand_px = tiles_width_px - map_width_px
            self.map = ImageOps.expand(self.map, border=(0, 0, right_expand_px, 0), fill=(0, 0, 0))
        if map_height_px != tiles_height_px:  # Expand the map bottom
            bottom_expand_px = tiles_height_px - map_height_px
            self.map = ImageOps.expand(self.map, border=(0, 0, 0, bottom_expand_px), fill=(0, 0, 0))

        # Write the new tiles to the map
        for (x_delta, y_delta), tile in self.tiles.items():
            if (x_delta, y_delta) in self.map_tiles:
                continue
            self.map.paste(
                tile,
                (
                    256 * (x_delta + tiles_top_left_x_delta),
                    256 * (y_delta + tiles_top_left_y_delta),
                ),
            )
            self.map_tiles.append((x_delta, y_delta))

        self.load_map_lock.release()
        return self.map

    def wait_tiles_loaded(self):
        """
        Waits until all tiles that were sent to load are loaded.
        """
        while self.num_tiles_loading > 0:
            time.sleep(0.1)

    def show(self, mono=False):
        curr_map = self.get_mono_map() if mono else self.get_map()
        if curr_map.size[0] != 0 and curr_map.size[1] != 0:
            curr_map.show()

    def size(self):
        return self.get_map().size

    def save(self, save_path):
        self.get_map().save(save_path)

    def get_tile_keys(self):
        return list(self.tiles.keys())

    def draw_roads(self, roads_lat_lon_deg):
        map_draw = ImageDraw.Draw(self.get_map())
        width_px, height_px = BaseMap.get_dimensions_px(self.tiles.keys())
        tiles_top_left_x_delta, tiles_top_left_y_delta = BaseMap.get_top_left_delta(self.tiles.keys())
        tile_x = self.origin_tile_x - tiles_top_left_x_delta
        tile_y = self.origin_tile_y - tiles_top_left_y_delta
        for street_name, road_edges_lat_lon_deg in roads_lat_lon_deg.items():
            for road_edge_lat_lon_deg in road_edges_lat_lon_deg:
                road_uv_px = [
                    tilemap_utils.get_px_rel_tile(lat_deg, lon_deg, tile_x, tile_y, self.zoom)
                    for lat_deg, lon_deg in road_edge_lat_lon_deg
                ]
                road_uv_px = geometry_utils.bound_line(road_uv_px, width_px, height_px)
                if len(road_uv_px) == 0:
                    continue
                map_draw.line(
                    [(int(x[0]), int(x[1])) for x in road_uv_px],
                    fill=color_utils.get_color(street_name),
                    width=8,
                )

    def get_px(self, lat_deg, lon_deg):
        tile_x, tile_y, _ = tilemap_utils.get_tile(lat_deg, lon_deg, self.zoom)
        x_px, y_px = tilemap_utils.get_px_on_tile(lat_deg, lon_deg, self.zoom)
        tiles_top_left_x_delta, tiles_top_left_y_delta = BaseMap.get_top_left_delta(self.tiles.keys())
        x_px += (tile_x - (self.origin_tile_x - tiles_top_left_x_delta)) * 256.0
        y_px += (tile_y - (self.origin_tile_y - tiles_top_left_y_delta)) * 256.0
        return (x_px, y_px)

    def get_lat_lon_bounds(self, tile_keys=None):
        """
        Returns the lat/lon in clockwise order, starting at the top left corner
        and ending at the bottom left corner.
        """
        if tile_keys is None:
            tile_keys = self.tiles.keys()
        tile_indexes = [
            (self.origin_tile_x + x_delta, self.origin_tile_y + y_delta) for (x_delta, y_delta) in tile_keys
        ]
        tile_xs, tile_ys = np.array(tile_indexes).T.tolist()
        top_left_lat_lon_deg = tilemap_utils.get_lat_lon_for_tile_px(min(tile_xs), min(tile_ys), 0.0, 0.0, self.zoom)
        top_right_lat_lon_deg = tilemap_utils.get_lat_lon_for_tile_px(max(tile_xs), min(tile_ys), 255.0, 0.0, self.zoom)
        bottom_right_lat_lon_deg = tilemap_utils.get_lat_lon_for_tile_px(
            max(tile_xs), max(tile_ys), 255.0, 255.0, self.zoom
        )
        bottom_left_lat_lon_deg = tilemap_utils.get_lat_lon_for_tile_px(
            min(tile_xs), max(tile_ys), 0.0, 255.0, self.zoom
        )
        return [
            top_left_lat_lon_deg,
            top_right_lat_lon_deg,
            bottom_right_lat_lon_deg,
            bottom_left_lat_lon_deg,
        ]

    def get_lat_lon_center(self):
        lat_lon_bounds = np.array(self.get_lat_lon_bounds())
        return [float(lat_lon_bounds[:, 0].mean()), float(lat_lon_bounds[:, 1].mean())]
