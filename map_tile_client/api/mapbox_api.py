import json
import mapbox_vector_tile
import requests

from map_tile_client.api._base_api import BaseAPI
from map_tile_client.models._base_map import BaseMap
from map_tile_client.utils import geometry_utils


class MapBoxAPI(BaseAPI):

    VTILE_BASE_URL = "https://api.mapbox.com/v4/mapbox.mapbox-streets-v8"

    def __init__(self):
        super().__init__()

    def get_access_token(self, reload=False):
        if not reload and hasattr(self, "access_token"):
            return self.access_token
        # resp = requests.post("https://build.symbium.com/lib/php/getmapboxkey.php",
        #                      data={"key": "build"})
        # self.access_token = resp.text
        self.access_token = "pk.eyJ1Ijoic3ltYml1bWRldiIsImEiOiJjazBpdXJiemcwM3p6M2JvM" \
                            "jA2bDA2cTJwIn0.aon9ZULViFwIj9Flt9vl7Q"
        return self.access_token

    def get_roads(self, curr_map):
        roads = {}
        tile_keys = curr_map.get_tile_keys()
        top_left_x_delta, top_left_y_delta = BaseMap.get_top_left_delta(tile_keys)
        for (x_delta, y_delta) in tile_keys:
            x, y = curr_map.origin_tile_x + x_delta, curr_map.origin_tile_y + y_delta
            mapbox_vtile_url = f"{MapBoxAPI.VTILE_BASE_URL}/{curr_map.zoom}/{x}/{y}.vector.pbf" \
                            f"?sku=101sHi7zTy5Qn&access_token={self.access_token}"
            curr_vtile = mapbox_vector_tile.decode(requests.get(mapbox_vtile_url).content)
            if "road" in curr_vtile:
                for feature in curr_vtile["road"]["features"]:
                    if (
                        feature["geometry"]["type"] != "LineString"
                        or feature["properties"]["class"] != "street"
                    ):
                        continue
                    curr_frame = [
                        tuple([256.0 * (float(point[0]) / 4096.0),
                            256.0 - (256.0 * (float(point[1]) / 4096.0))])
                        for point in feature["geometry"]["coordinates"]
                    ]
                    name = feature["properties"]["name"] if "name" in feature["properties"] \
                                                        else feature["properties"]["type"]
                    if name not in roads:
                        roads[name] = []
                    curr_segment = [
                        (point[0] + (256.0 * (x_delta + top_left_x_delta)),
                         point[1] + (256.0 * (y_delta + top_left_y_delta)))
                        for point in curr_frame
                    ]
                    roads[name] = geometry_utils.merge_lines([roads[name], curr_segment])
        return roads
