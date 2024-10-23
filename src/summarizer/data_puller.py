from datetime import datetime
from src.rag.shared.database import execute_query, generate_database_connection
from dotenv import load_dotenv
from pathlib import Path
from os import getenv
from typing import Dict
import pandas as pd
from src.rag.shared.cloud_storage import BackBlazeCloudStorage
from tempfile import NamedTemporaryFile


class DataPuller:

    """ This class will be responsible to read the data for the new summarizer and save the data into a local storage or into a cloud bucket."""

    def __init__(self, environment: str) -> None:

        current_directory = Path.cwd()
        env_file = current_directory.joinpath(f"env_{environment}")
        self.current_directory = current_directory
        self.environment = environment

        load_dotenv(dotenv_path=env_file)

    def load_database_credentials(self) -> Dict[str, str]:
        """ Load an environment variables and return them, raise a value error if one of them is empty."""
        database_user = getenv('POSTGRES_USER')
        database_password = getenv('POSTGRES_PASSWORD')
        database_host = getenv('POSTGRES_HOST')
        database_port = getenv('POSTGRES_PORT')
        database_name = getenv('POSTGRES_DB')

        if not database_user or not database_password or not database_host or not database_port or not database_name:
            raise ValueError("Database credentials are missing")

        return {
            'user': database_user,
            'password': database_password,
            'host': database_host,
            'port': database_port,
            'database': database_name
        }

    def read_data(self, date: str) -> pd.DataFrame:
        """ Read the data from the database and return the pnadas dataframe of the data.    """

        database_credentials = self.load_database_credentials()

        connection = generate_database_connection(database_credentials)
        article_query = "select content, title, posted_at,url from article where posted_at::date = '{}'".format(
            date)
        today_articles = execute_query(connection, article_query)
        news_df = pd.DataFrame(today_articles)
        news_df.columns = ["content", "title", "posted_at", "url"]
        news_df = news_df.drop_duplicates(
            subset="content").reset_index(drop=True)

        return news_df

    def save_data(self, data: pd.DataFrame) -> None:
        """ Save data either to the local environment or to a cloud bucket."""

        pass

    def save_local(self, data: pd.DataFrame) -> None:
        """ Save data to the local environment."""
        today = datetime.now().strftime("%Y-%m-%d")
        news_directory = self.current_directory.joinpath(
            "datasets", "today_news")
        data.to_csv(news_directory.joinpath(f"{today}-news.csv"))

    def save_to_blackbaze_bucket(self, data: pd.DataFrame) -> None:
        """ Save data to a cloud bucket."""
        cloud_storage = BackBlazeCloudStorage(environment=self.environment)
        with NamedTemporaryFile(delete=True) as temp_file:
            data.to_csv(temp_file)
            cloud_storage.upload_file(
                bucket_name="news",
                file_path=temp_file,
                file_name=temp_file.name,
                metadata={"date": datetime.now().strftime("%Y-%m-%d")}
            )
