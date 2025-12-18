# fetchers/steam_store_fetcher.py
from .base import Fetcher
import requests
from typing import List
from logs.logger import get_logger
import os
import json
from ratelimit import limits, sleep_and_retry
from utils.request_rate_limiter import rate_limited_request
from utils.steam_http import steam_get

# allow 100 requests per minute
ONE_MINUTE=60

class SteamStoreFetcher(Fetcher):
    """Fetch Steam Store metadata for a list of app IDs."""
    def __init__(self, context, batch_size: int = 50):
        super().__init__(context)
        self.logger = get_logger(self.__class__.__name__)
        self.batch_size = batch_size
        self.logger.info("Initialized SteamStoreFetcher")

    def fetch(self, app_ids: List[int], cc="ie", l="en") -> list:
        results = []

        for i, app_id in enumerate(app_ids):
            self.logger.info(f"Fetching store data for AppID {app_id}")

            response = steam_get(
                "https://store.steampowered.com/api/appdetails",
                params={"appids": app_id, "cc": cc, "l": l},
                timeout=self.context.timeout
            )
            if not response:
                self.logger.warning(f"Failed to fetch data for AppID {app_id}")
                continue

            payload = response.json().get(str(app_id), {})
            if not payload.get("success"):
                continue

            data = payload["data"]

            if i == 0:
                os.makedirs("debug", exist_ok=True)
                with open("debug/steam_store_payload_example.json", "w", encoding="utf-8") as f:
                    json.dump(data, f, indent=2, ensure_ascii=False)

            results.append(data)

        self.logger.info(f"Fetched store data for {len(results)} apps")
        return results

    # @sleep_and_retry
    # @limits(calls=100, period=ONE_MINUTE)  # 100 requests/min
    # def fetch_app_batch(self, app_ids: List[int], cc: str = "ie", l: str = "en") -> List[dict]:
    #     results = []
    #     url = "https://store.steampowered.com/api/appdetails"
    #     for app_id in app_ids:
    #         self.logger.info(f"Fetching data for AppID: {app_id}")
    #         params = {"appids": app_id, "cc": cc, "l": l}
    #
    #         try:
    #             response = requests.get(url, params=params, timeout=self.context.timeout)
    #             response.raise_for_status()
    #         except requests.RequestException as e:
    #             self.logger.warning(f"Failed to fetch {app_id}: {e}")
    #             continue
    #
    #         payload = response.json().get(str(app_id), {})
    #         if payload.get("success"):
    #             results.append(payload["data"])
    #
    #     return results
    # def fetch_app_batch(self, app_ids: list[int], batch_size=50) -> list:
    #     results = []
    #     for i in range(0, len(app_ids), batch_size):
    #         batch = app_ids[i:i + batch_size]
    #         url = "https://store.steampowered.com/api/appdetails"
    #         params = {"appids": ",".join(map(str, batch)), "cc": "ie", "l": "en"}
    #
    #         data = rate_limited_request(url, params=params, timeout=self.context.timeout).json()
    #         for app_id_str, app_data in data.items():
    #             if app_data.get("success"):
    #                 results.append(app_data["data"])
    #     return results
    #
    # def fetch(self, app_ids: List[int], cc: str = "ie", l: str = "en") -> list:
    #     results = []
    #
    #     for i, app_id in enumerate(app_ids):
    #         self.logger.info(f"Fetching data for AppID: {app_id}")
    #         url = f"https://store.steampowered.com/api/appdetails"
    #         params = {"appids": app_id
    #                   , "cc": cc
    #                   , "l": l
    #                   }
    #
    #         response = requests.get(
    #             url,
    #             params=params,
    #             timeout=self.context.timeout
    #         )
    #
    #         if response.status_code != 200:
    #             continue
    #
    #         payload = response.json().get(str(app_id), {})
    #         if not payload.get("success"):
    #             continue
    #
    #         data = payload["data"]
    #
    #         # 🔍 SAVE FIRST PAYLOAD ONLY
    #         if i == 0:
    #             os.makedirs("debug", exist_ok=True)
    #             with open(
    #                     "debug/steam_store_payload_example.json",
    #                     "w",
    #                     encoding="utf-8"
    #             ) as f:
    #                 json.dump(data, f, indent=2, ensure_ascii=False)
    #
    #             self.logger.info(
    #                 "Saved example Steam Store payload to debug/steam_store_payload_example.json"
    #             )
    #
    #         results.append(data)
    #     self.logger.info(f"Fetched data for {len(results)} out of {len(app_ids)} AppIDs")
    #     return results

