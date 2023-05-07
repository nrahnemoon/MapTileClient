"""
Util functions to convert lat/lon/zoom to tile x/y/zoom and vice/versa.
"""
import math
import numpy as np
from scipy.constants import degree


def get_point(latitude_deg, longitude_deg):
    """
    Returns (x, y) point in global coordinates as float values, assuming
    the entire world is 256 by 256 (x/y can be 0 min and 256.0 max).

    Parameters:
        latitude_deg (float): latitude in degrees,
        longitude_deg (float): longitude in degrees.

    Returns:
        (
            (float): the x coodinate,
            (float): the y coodinate
        )
    """
    siny = min(max(math.sin(latitude_deg * degree), -0.9999), 0.9999)
    x = 128 + longitude_deg * (256 / 360)
    y = 128 + 0.5 * math.log((1 + siny) / (1 - siny)) * -(256 / (2 * np.pi))
    return x, y


def get_tile(latitude_deg, longitude_deg, zoom):
    """
    Given a lat/lon/zoom value, return the tile x/y/zoom.
    Read more about this here: https://docs.microsoft.com/en-us/azure/azure-maps/zoom-levels-and-tile-grid

    Parameters:
        latitude_deg (float): the latitude of the point to convert, in degrees,
        longitude_deg (float): the longitude of the point to convert, in degrees,
        zoom (int): the zoom level

    Returns:
        (
            (int): the x coordinate for a given zoom level,
            (int): the y coordinate for a given zoom level,
            (zoom): the zoom level
        )
    """
    tile_divisions_per_axis = math.pow(2, zoom)
    width_of_tile = 256 / tile_divisions_per_axis
    x, y = get_point(latitude_deg, longitude_deg)
    x = math.floor(x / width_of_tile)
    y = math.floor(y / width_of_tile)
    return x, y, zoom


def get_px_on_tile(latitude_deg, longitude_deg, zoom):
    """
    Given a lat/lon/zoom, return the x, y pixel of the location on the tile.

    Parameters:
        latitude_deg (float): the latitude of the point to get the px of,
        longitude_deg (float): the longitude of the point to get the px of,
        zoom (int): the zoom level.

    Returns:
        (
            (float): pixel location along the width axis of the tile (max 256.0),
            (float): pixel location along the height axis of the tile (max 256.0)
        )
    """
    tile_divisions_per_axis = math.pow(2, zoom)
    width_of_tile = 256 / tile_divisions_per_axis
    x, y = get_point(latitude_deg, longitude_deg)
    x = (x / width_of_tile % 1.0) * 255
    y = (y / width_of_tile % 1.0) * 255
    return x, y


def get_px_rel_tile(latitude_deg, longitude_deg, tile_x, tile_y, zoom):
    """
    Given a lat/lon/zoom and a tile_x/tile_y, the x, y pixel of the location relative to the tile.

    Parameters:
        latitude_deg (float): the latitude of the point to get the px of,
        longitude_deg (float): the longitude of the point to get the px of,
        zoom (int): the zoom level.

    Returns:
        (
            (float): pixel location along the width axis of the tile,
            (float): pixel location along the height axis of the tile
        )
    """
    orig_tile_x, orig_tile_y, _ = get_tile(latitude_deg, longitude_deg, zoom)
    x, y = get_px_on_tile(latitude_deg, longitude_deg, zoom)
    x += (orig_tile_x - tile_x) * 256.0
    y += (orig_tile_y - tile_y) * 256.0
    return x, y


def get_lat_lon_from_point(point_x, point_y):
    """
    Opposite function to `get_point`.
    Given a point in world x/y coordinates, returns the corresponding lat/lon.

    Parameters:
        point_x (float): the x coordinate of the point in world x/y coordinates,
        point_y (float): the y coordinate of the point in world x/y coordinates

    Returns:
        (
            (float): the latitude of the point in degrees,
            (float): the longitude of the point in degrees
        )
    """
    latitude_deg = (2 * math.atan(math.exp((point_y - 128) / -(256 / (2 * np.pi)))) - np.pi / 2) / degree
    longitude_deg = (point_x - 128) / (256 / 360)
    return latitude_deg, longitude_deg


def get_lat_lon_for_tile_px(tile_x, tile_y, x_px, y_px, zoom):
    """
    Given a pixel location (x_px, y_px) on a specific tile (described by tile_x, tile_y, zoom),
    returns the lat/lon corresponding to that point.

    Parameters:
        tile_x (int): the x coordinate of the tile,
        tile_x (int): the y coordinate of the tile,
        x_px (float): the pixel location of the tile along the width dimension,
        y_px (float): the pixel location of the tile along the height dimension,
        zoom (int): the zoom level of the tile

    Returns:
        (
            (float): the latitude of the described point, in degrees,
            (float): the longitude of the described point, in degrees
        )
    """
    tile_divisions_per_axis = math.pow(2, zoom)
    width_of_tile = 256 / tile_divisions_per_axis
    x = (tile_x + (x_px / 255)) * width_of_tile
    y = (tile_y + (y_px / 255)) * width_of_tile
    return get_lat_lon_from_point(x, y)
