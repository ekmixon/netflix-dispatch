import json
import time

from jose import jwt
import requests


API_BASE_URI = "https://api.zoom.us/v2"


class ZoomClient:
    """Simple HTTP Client for Zoom Calls."""

    def __init__(self, api_key, api_secret):
        self.api_key = api_key
        self.api_secret = api_secret
        self.token = generate_jwt(self.api_key, self.api_secret)
        self.headers = self._get_headers()
        self.timeout = 15

    def _get_headers(self):
        return {
            "authorization": f"Bearer {self.token}",
            "content-type": "application/json",
        }

    def get(self, path, params=None):
        return requests.get(
            f"{API_BASE_URI}/{path}",
            params=params,
            headers=self.headers,
            timeout=self.timeout,
        )

    def post(self, path, data):
        return requests.post(
            f"{API_BASE_URI}/{path}",
            data=json.dumps(data),
            headers=self.headers,
            timeout=self.timeout,
        )

    def delete(self, path, data=None, params=None):
        return requests.delete(
            f"{API_BASE_URI}/{path}",
            data=json.dumps(data),
            params=params,
            headers=self.headers,
            timeout=self.timeout,
        )


def generate_jwt(key, secret):
    header = {"alg": "HS256", "typ": "JWT"}
    payload = {"iss": key, "exp": int(time.time() + 3600)}
    return jwt.encode(payload, secret, algorithm="HS256", headers=header)
