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
        bucket_config = {
            "endpoint": endpoint,
            "ak": user_config["ak"],
            "sk": user_config["sk"],
        }
        bucket_configs[bucket_name] = bucket_config

    return bucket_configs


def write_json_to_home(my_dict):
    # Convert dictionary to JSON
    json_data = json.dumps(my_dict, indent=4, ensure_ascii=False)

    # Determine the home directory path based on the operating system
    if os.name == "posix":  # Linux or macOS
        home_dir = os.path.expanduser("~")
    elif os.name == "nt":  # Windows
        home_dir = os.path.expandvars("%USERPROFILE%")
    else:
        raise Exception("Unsupported operating system")

    # Define the output file path
    output_file = os.path.join(home_dir, "magic_pdf_config.json")

    # Write JSON data to the output file
    with open(output_file, "w") as f:
        f.write(json_data)

    # Print a success message
    print(f"Dictionary converted to JSON and saved to {output_file}")


if __name__ == '__main__':
    bucket_configs_dict = get_bucket_configs_dict(s3_buckets, s3_clusters, s3_users)
    logger.info(bucket_configs_dict)
    write_json_to_home(bucket_configs_dict)
