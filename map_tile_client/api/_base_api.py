"""
Abstract class to provide generic API load and processing functionality.
"""
import abc
import json
import functools


class BaseAPI:
    NUM_RETRY_ATTEMPTS = 2

    def __init__(self):
        self.get_access_token()

    @abc.abstractmethod
    def get_access_token(self, reload=True):
        raise NotImplementedError

    @staticmethod
    def process_response(response):
        """
        Parses the API response.  If the HTTP status isn't 200, raises an exception.

        Parameters:
            response (requests.models.Response): the response object returned
                                                 by requests.get or requests.post.

        Returns:
            (dict): the response in dictionary format
        """
        if response.status_code != 200:
            raise Exception("Errror hitting API.\n" f"\tStatus code: {response.status_code},\n\tData: {response.text}")
        return json.load(response.text)

    def retry(function):
        @functools.wraps(function)
        def retry_function(*args, **kwargs):
            for attempt in range(1, BaseAPI.NUM_RETRY_ATTEMPTS + 1):  # 2 attempts
                try:
                    return function(*args, **kwargs)
                except Exception as e:
                    args[0].get_access_token(reload=True)
                    print(f"[{attempt}/{BaseAPI.NUM_RETRY_ATTEMPTS}] Call failed, retrying." f"\n\tException: {e}")

        return retry_function
