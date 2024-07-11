import yaml
from b2sdk.v2 import B2Api, InMemoryAccountInfo

# Load configuration
config = yaml.safe_load(open("config.yaml"))

# Initialize B2 API
info = InMemoryAccountInfo()
b2_api = B2Api(info)

# Authorize with your application key ID and application key
application_key_id = config["b2_read_key_id"]
application_key = config["b2_read_key"]
b2_api.authorize_account("production", application_key_id, application_key)

# Bucket and file details
bucket_name = config["b2_bucket_name"]
file_name = "n01440764_18.JPEG"  # The file you want to download
download_path = "./n01440764_18.JPEG"  # Modify as needed

# Get the bucket
bucket = b2_api.get_bucket_by_name(bucket_name)

# Download file
file_version = bucket.download_file_by_name(file_name)
with open(download_path, "wb") as file:
    file_version.save(file)

print(f"File downloaded: {download_path}")
