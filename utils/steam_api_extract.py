from pipelines.steam_pipeline import SteamDataPipeline
from logs.logger import get_logger
from dotenv import load_dotenv
import os


if __name__ == "__main__":
    # from data.steam_extractor import SteamAPIExtractor
    # from data.abstract_connector import DBConnector
    # from data.steam_api_client import SteamAPIClient
    #
    # # Initialize database connector
    # connector = DBConnector(
    #     server="your_server",
    #     database="your_database",
    #     username="your_username",
    #     password="your_password"
    # )
    #
    # # Initialize Steam API client
    # api_client = SteamAPIClient(api_key="your_steam_api_key")
    #
    # # Fetch data from Steam API
    # steam_games = api_client.fetch_all_games()
    #
    # # Initialize extractor
    # extractor = SteamAPIExtractor(connector=connector, chunk_size=100)
    #
    # # Save data to database
    # extractor.save_data(steam_games)
    load_dotenv()
    logger = get_logger("steam_api_extract")
    api_key = os.getenv("STEAM_API_KEY")
    if not api_key:
        logger.error("STEAM_API_KEY not found in environment variables.")
        exit(1)

    logger.info("Starting Steam data extraction pipeline...")

    steam_pipeline = SteamDataPipeline(
        api_key=api_key,
        chunk_size=100,
        limit_apps=10000,  # Limit to first 10 apps for testing
        offset_apps=0
    )
    logger.info("Running Steam data extraction pipeline...")
    steam_pipeline.run()
    logger.info("Steam data extraction pipeline completed.")
