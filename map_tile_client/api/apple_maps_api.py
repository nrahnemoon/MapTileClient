import json
import requests

from map_tile_client.api._base_api import BaseAPI


class AppleMapsAPI(BaseAPI):

    def __init__(self):
        super().__init__()

    def get_access_token(self, reload=False):
        if not reload and hasattr(self, "access_token"):
            return self.access_token
        bearer = requests.get("https://duckduckgo.com/local.js?get_mk_token=1").text
        headers = {"authorization": f"Bearer {bearer}"}
        resp = requests.get("https://cdn.apple-mapkit.com/ma/bootstrap", headers=headers)
        self.access_token = json.loads(resp.text)["tileSources"][0]["path"].split("&accessKey=")[1]
        return self.access_token
