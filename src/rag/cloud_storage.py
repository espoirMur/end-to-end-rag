import os
import b2sdk.v2 as back_blaze
from dotenv import load_dotenv


class BackBlazeCloudStorage:
    """
    This class is responsible for interacting with backblaze cloud storage.

    It will instanciate the cloud storage client and provide methods for uploading and downloading files
    """

    def load_environment_variables(self):
        """ Load an environment variables and return them, raise a value error if one of them is empty."""
        load_dotenv()
        application_key_id = os.getenv("BACK_BLAZE_KEY_ID")
        application_key = os.getenv("BACK_BLAZE_APPLICATION_KEY")

        if not application_key_id or not application_key:
            raise ValueError("Application key id or application key is empty")

        return application_key_id, application_key

    def __init__(self) -> None:
        application_key_id, application_key = self.load_environment_variables()
        info = back_blaze.InMemoryAccountInfo()
        self.back_blaze_api = back_blaze.B2Api(info)
        self.back_blaze_api.authorize_account(
            "production", application_key_id, application_key)

    def get_bucket(self, bucket_name: str):
        return self.back_blaze_api.get_bucket_by_name(bucket_name)

    def upload_file(self, bucket_name: str, file_path: str, file_name: str, metadata: dict) -> str:
        """ upload the file to the bucket with the given name"""
        bucket = self.get_bucket(bucket_name)
        uploaded_file = bucket.upload_local_file(local_file=file_path,
                                                 file_name=file_name, file_infos=metadata)
        return self.back_blaze_api.get_download_url_for_fileid(uploaded_file.id_)