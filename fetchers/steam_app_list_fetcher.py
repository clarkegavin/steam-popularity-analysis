# fetchers/steam_app_list_fetcher.py
from .base import Fetcher
import requests
from logs.logger import get_logger
from utils.steam_http import steam_get
import os
import json
from ratelimit import limits, sleep_and_retry
from typing import Dict

ONE_MINUTE = 60

class SteamAppListFetcher(Fetcher):
    """Fetch all Steam AppIDs."""
    def __init__(self, context):
        super().__init__(context)
        self.logger = get_logger(self.__class__.__name__)

    # def fetch(self) -> dict:
    #     url = "https://api.steampowered.com/IStoreService/GetAppList/v1/"
    #     params = {"key": self.context.api_key,
    #               "include_games": 1
    #               } if self.context.api_key else {}
    #     self.logger.info(f"Fetching Steam app list from API with params: {params}")
    #
    #     response = requests.get(url, params=params, timeout=self.context.timeout)
    #     response.raise_for_status()
    #     payload = response.json()
    #     apps = payload.get("response", {}).get("apps", [])
    #     self.logger.info(f"Retrieved {len(apps)} Steam apps")
    #     # Return the 'apps' list
    #     return {
    #         app["appid"]: app.get("name", "")
    #         for app in apps
    #     }

    # @sleep_and_retry
    # @limits(calls=100, period=ONE_MINUTE)  # ~100 requests per minute
    # def fetch_app_details(self, app_id: int) -> Dict:
    #     url = "https://store.steampowered.com/api/appdetails"
    #     params = {"appids": app_id, "cc": "ie", "l": "en"}
    #     try:
    #         resp = requests.get(url, params=params, timeout=self.context.timeout)
    #         resp.raise_for_status()
    #         payload = resp.json().get(str(app_id), {})
    #         if payload.get("success"):
    #             return payload["data"]
    #     except Exception as e:
    #         self.logger.warning(f"Failed to fetch AppID {app_id}: {e}")
    #     return {}
    #
    # def fetch(self, start_id: int = 0, batch_size: int = 100, max_empty_batches: int = 50) -> Dict[int, Dict]:
    #     """Fetch all Steam apps by iterating AppID ranges."""
    #     self.logger.info("Starting to fetch all Steam apps by AppID ranges")
    #     all_apps = {}
    #     empty_batches = 0
    #     current_id = start_id
    #
    #     while empty_batches < max_empty_batches:
    #         batch_apps = []
    #         for app_id in range(current_id, current_id + batch_size):
    #             self.logger.debug(f"Fetching AppID: {app_id}")
    #             data = self.fetch_app_details(app_id)
    #             if data:
    #                 all_apps[app_id] = data
    #                 batch_apps.append(app_id)
    #
    #         if batch_apps:
    #             empty_batches = 0
    #             self.logger.info(
    #                 f"Fetched {len(batch_apps)} apps for AppID range {current_id}-{current_id + batch_size - 1}")
    #         else:
    #             empty_batches += 1
    #             self.logger.info(
    #                 f"No apps found for AppID range {current_id}-{current_id + batch_size - 1} ({empty_batches}/{max_empty_batches})")
    #
    #         current_id += batch_size
    #
    #     self.logger.info(f"Finished fetching apps. Total apps retrieved: {len(all_apps)}")
    #
    #     # Save debug payload for first app
    #     if all_apps:
    #         os.makedirs("debug", exist_ok=True)
    #         first_app_id = list(all_apps.keys())[0]
    #         with open("debug/steam_app_list_first_app.json", "w", encoding="utf-8") as f:
    #             json.dump(all_apps[first_app_id], f, indent=2, ensure_ascii=False)
    #         self.logger.info(f"Saved first app payload to debug/steam_app_list_first_app.json")
    #
    #     return all_apps

    def fetch(self) -> dict:
        url = "https://api.steampowered.com/IStoreService/GetAppList/v1/"
        params = {
            "key": self.context.api_key,
            "include_games": 1,
            "cc": "ie",
            "l": "en"
        }

        response = steam_get(url, params, timeout=self.context.timeout)
        apps = response.json().get("response", {}).get("apps", [])

        self.logger.info(f"Retrieved {len(apps)} Steam apps")
        return {app["appid"]: app["name"] for app in apps}