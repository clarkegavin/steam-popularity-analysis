# fetchers/steam_review_fetcher.py
from .base import Fetcher
import requests
from typing import List
from logs.logger import get_logger
import os
import json
from utils.steam_http import steam_get
from ratelimit import limits, sleep_and_retry
from utils.request_rate_limiter import rate_limited_request

ONE_MINUTE = 60

class SteamReviewFetcher(Fetcher):
    """Fetch Steam review stats via SteamSpy."""
    def __init__(self, context):
        super().__init__(context)
        self.logger = get_logger(self.__class__.__name__)
        self.logger.info("Initialized SteamReviewFetcher")

    def fetch(self, app_ids: List[int], cc="ie", language="english") -> list:
        results = []

        for i, app_id in enumerate(app_ids):
            self.logger.info(f"Fetching review summary for AppID {app_id}")
            url = f"https://store.steampowered.com/appreviews/{app_id}"
            params = {
                "json": 1,
                "language": language,
                "purchase_type": "all",
                "num_per_page": 1
            }

            try:
                response = steam_get(url, params=params, timeout=self.context.timeout)
                # response = steam_get(
                #     f"https://store.steampowered.com/appreviews/{app_id}",
                #     params={
                #         "json": 1,
                #         "language": language,
                #         "purchase_type": "all",
                #         "num_per_page": 1
                #     },
                #     timeout=self.context.timeout
                # )
                if response is None:
                    self.logger.warning(f"Failed to fetch reviews for AppID {app_id}")
                    continue
                payload = response.json()
            except Exception as e:
                self.logger.warning(f"Exception fetching reviews for AppID {app_id}: {e}")
                continue


            summary = payload.get("query_summary")
            if not summary:
                continue

            if i == 0:
                os.makedirs("debug", exist_ok=True)
                with open("debug/steam_review_payload_example.json", "w", encoding="utf-8") as f:
                    json.dump(payload, f, indent=2, ensure_ascii=False)

           #self.logger.info(f"Review summary for AppID {app_id}: {summary}, review_score: {summary.get('review_score')}, total_reviews: {summary.get('total_reviews')}")
            results.append({
                "AppID": app_id,
                "review_score": summary.get("review_score"),
                "review_score_desc": summary.get("review_score_desc"),
                "total_reviews": summary.get("total_reviews"),
                "total_positive": summary.get("total_positive"),
                "total_negative": summary.get("total_negative"),
            })

        self.logger.info(f"Fetched review summaries for {len(results)} apps")
        return results
    # def fetch(self, app_ids: List[int]) -> dict:
    #     data = {}
    #     for appid in app_ids:
    #         url = f"https://steamspy.com/api.php?request=appdetails&appid={appid}"
    #         r = requests.get(url)
    #         r.raise_for_status()
    #         data[appid] = r.json()
    #     return data

    # @sleep_and_retry
    # @limits(calls=100, period=ONE_MINUTE)
    # def fetch_review_summary(self, app_id: int, cc: str = "ie", l: str = "en") -> dict:
    #     url = f"https://store.steampowered.com/appreviews/{app_id}"
    #     params = {
    #         "json": 1,
    #         "language": l,
    #         "purchase_type": "all",
    #         "filter": "all",
    #         "cc": cc,
    #         "num_per_page": 1  # only fetch summary info, not individual reviews
    #     }

        # try:
        #     response = requests.get(url, params=params, timeout=self.context.timeout)
        #     response.raise_for_status()
        # except requests.RequestException as e:
        #     self.logger.warning(f"Failed to fetch reviews for {app_id}: {e}")
        #     return {}
        # response = rate_limited_request(url, params=params, timeout=self.context.timeout)
        # payload = response.json()
        # summary = {
        #     "AppID": app_id,
        #     "Review_Score": payload.get("query_summary", {}).get("review_score"),
        #     "Review_Score_Desc": payload.get("query_summary", {}).get("review_score_desc"),
        #     "Total_Reviews": payload.get("query_summary", {}).get("total_reviews"),
        #     "Total_Positive": payload.get("query_summary", {}).get("total_positive"),
        #     "Total_Negative": payload.get("query_summary", {}).get("total_negative"),
        # }
        #
        # # Save first payload for debug
        # if app_id == 0:
        #     os.makedirs("debug", exist_ok=True)
        #     with open("debug/steam_review_payload_example.json", "w", encoding="utf-8") as f:
        #         json.dump(payload, f, indent=2, ensure_ascii=False)
        #     self.logger.info("Saved example Steam Review payload to debug/steam_review_payload_example.json")
        #
        # return summary

    # def fetch(self, app_ids: List[int], language: str = "all", review_type: str = "all") -> list:
    #     """Fetch reviews for given app IDs.
    #
    #     Parameters
    #     ----------
    #     app_ids : list[int]
    #         Steam App IDs
    #     language : str
    #         Review language filter ("all", "english", etc.)
    #     review_type : str
    #         "all", "positive", or "negative"
    #
    #     Returns
    #     -------
    #     list
    #         List of review payloads
    #     """
    #     results = []
    #
    #     for i, app_id in enumerate(app_ids):
    #         self.logger.info(f"Fetching reviews for AppID: {app_id}")
    #         url = f"https://store.steampowered.com/appreviews/{app_id}"
    #         params = {
    #             "json": 1,
    #             "language": language,
    #             "filter": "all",
    #             "review_type": review_type,
    #             "purchase_type": "all",
    #             "num_per_page": 100  # adjust if needed
    #         }
    #
    #         try:
    #             response = requests.get(url, params=params, timeout=self.context.timeout)
    #             response.raise_for_status()
    #         except requests.RequestException as e:
    #             self.logger.warning(f"Failed to fetch reviews for {app_id}: {e}")
    #             continue
    #
    #         data = response.json()
    #
    #         # Save first payload only
    #         if i == 0:
    #             os.makedirs("debug", exist_ok=True)
    #             example_file = "debug/steam_review_payload_example.json"
    #             with open(example_file, "w", encoding="utf-8") as f:
    #                 json.dump(data, f, indent=2, ensure_ascii=False)
    #             self.logger.info(f"Saved example review payload to {example_file}")
    #
    #         results.append(data)
    #
    #     self.logger.info(f"Fetched review data for {len(results)} out of {len(app_ids)} AppIDs")
    #     return results