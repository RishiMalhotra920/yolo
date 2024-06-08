from b2sdk.v2 import InMemoryAccountInfo, B2Api
import yaml
config = yaml.safe_load(open("config.yaml"))
# Initialize API
info = InMemoryAccountInfo()
b2_api = B2Api(info)

# Authorize with your application key ID and application key
application_key_id = config["b2_write_key_id"]
application_key = config["b2_write_key"]
b2_api.authorize_account("production", application_key_id, application_key)

# File to upload and bucket details
bucket_name = config["b2_bucket_name"]
# file_path = '/Users/rishimalhotra/projects/cv/image_classification/image_net_data.tar.gz'
# file_name = 'image_net_data.tar.gz'
file_path = '/Users/rishimalhotra/projects/cv/image_classification/image_net_data/train/n01440764/n01440764_18.JPEG'
file_name = 'n01440764_18.JPEG'

# Get the bucket
bucket = b2_api.get_bucket_by_name(bucket_name)

# Upload file
b2_file = bucket.upload_local_file(
    local_file=file_path,
    file_name=file_name,
    content_type='application/gzip'
)

print(f"File uploaded: {b2_file.file_name}, ID: {b2_file.id_}")
