from loguru import logger
import json
import os
from magic_pdf.config import s3_buckets, s3_clusters, s3_users


def get_bucket_configs_dict(buckets, clusters, users):
    bucket_configs = {}
    for s3_bucket in buckets.items():
        bucket_name = s3_bucket[0]
        bucket_config = s3_bucket[1]
        cluster, user = bucket_config
        cluster_config = clusters[cluster]
        endpoint_key = "outside"
        endpoints = cluster_config[endpoint_key]
        endpoint = endpoints[0]
        user_config = users[user]
        # logger.info(bucket_name)
        # logger.info(endpoint)
        # logger.info(user_config)
        bucket_config = [user_config["ak"], user_config["sk"], endpoint]
        bucket_configs[bucket_name] = bucket_config

    return bucket_configs


def write_json_to_home(my_dict):
    # Convert dictionary to JSON
    json_data = json.dumps(my_dict, indent=4, ensure_ascii=False)

    home_dir = os.path.expanduser("~")

    # Define the output file path
    output_file = os.path.join(home_dir, "magic-pdf.json")

    # Write JSON data to the output file
    with open(output_file, "w") as f:
        f.write(json_data)

    # Print a success message
    print(f"Dictionary converted to JSON and saved to {output_file}")


if __name__ == '__main__':
    bucket_configs_dict = get_bucket_configs_dict(s3_buckets, s3_clusters, s3_users)
    logger.info(bucket_configs_dict)
    config_dict = {
        "bucket_info": bucket_configs_dict,
        "temp-output-dir": "/tmp"
    }
    write_json_to_home(config_dict)
