#!/usr/bin/env python3
"""
Standalone script to download VR dataset tar.gz files from cloud.
This script downloads tar.gz files from the datacloud service.

Requirements:
    pip install -r download_requirements.txt

Usage:
    python download_vr_dataset_cache.py --datacloud_endpoint http://10.128.1.42:8083 --identifier test --output_dir ./data

The script will download and extract tar.gz files to the specified output directory.
"""

import os
import tempfile
import hashlib
import json
import requests
import tarfile
import logging
import sys
import argparse
import lz4.frame

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def download_vr_dataset_tar_gz(
    datacloud_endpoint: str,
    identifier: str,
    query_filter: dict,
    output_dir: str
) -> str:
    """
    Download tar.gz files from cloud and extract to output directory.

    Returns the path to the output directory.
    """
    print(f"[Download] Starting download for identifier: {identifier}")

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Step 1: Get records from cloud
    print("[Download] Step 1: Get records from cloud")
    list_recordings_request = {
        "identifier": identifier,
        "query_filter": query_filter,
        "limit": 10000,
        "skip": 0,
    }
    url = f"{datacloud_endpoint}/v1/logs"
    print(f"[Download] Requesting records from {url}")

    try:
        response = requests.post(
            url,
            json=list_recordings_request,
            headers={"Content-Type": "application/json"}
        )
    except Exception as e:
        print(f"[Download] Request failed: {str(e)}")
        raise RuntimeError(f"Failed to connect to datacloud service: {e}")

    records = response.json().get('data', [])
    print(f"[Download] Received {len(records)} records from cloud")

    if len(records) == 0:
        raise RuntimeError(f"No records found for identifier '{identifier}' with query filter: {query_filter}")

    cloud_uuid_list = [record['uuid'] for record in records]
    logger.info(f"Found {len(cloud_uuid_list)} records in the cloud for identifier '{identifier}' with query filter: {query_filter}.")

    # Step 2: Download data from cloud
    print("[Download] Step 2: Download data from cloud")
    filename = os.path.join(output_dir, "downloaded_records.tar.gz")
    try:
        data_request = {
            "identifier": identifier,
            "uuids": cloud_uuid_list,
        }
        response = requests.post(
            f"{datacloud_endpoint}/v1/download_records",
            json=data_request,
            stream=True
        )

        total_size = int(response.headers.get('content-length', 0))
        print(f"[Download] Downloading {total_size} bytes from cloud")

        downloaded_size = 0
        if response.status_code == 200:
            with open(filename, 'wb') as f:
                for chunk in response.iter_content(chunk_size=1024*1024):
                    if chunk:
                        f.write(chunk)
                        downloaded_size += len(chunk)
                        print(f"\r[Download] Downloaded {downloaded_size}/{total_size} bytes", end="", flush=True)

            print()  # New line

            server_sha256sum = response.headers.get('X-File-SHA256')
            if server_sha256sum:
                sha256_hash = hashlib.sha256()
                with open(filename, 'rb') as f:
                    for chunk in iter(lambda: f.read(4096), b""):
                        sha256_hash.update(chunk)
                file_sha256sum = sha256_hash.hexdigest()
                if file_sha256sum != server_sha256sum:
                    logger.error(f"SHA256 checksum mismatch: {file_sha256sum} != {server_sha256sum}")
                    raise RuntimeError(f"SHA256 checksum mismatch: {file_sha256sum} != {server_sha256sum}")
                else:
                    print(f"[Download] SHA256 verification successful: {server_sha256sum}")
                    logger.info(f"Downloaded file SHA256: {server_sha256sum}, verification successful.")
            else:
                print("[Download] Warning: SHA256 checksum not provided")
                logger.warning("SHA256 checksum not provided in response headers.")
        else:
            raise RuntimeError(f"Download failed with status code {response.status_code}: {response.text}")
    except Exception as e:
        logger.error(f"Failed to download data from cloud: {str(e)}")
        raise RuntimeError(f"Failed to download data: {e}")

    # Step 3: Extract downloaded data
    print("[Download] Step 3: Extract downloaded data")
    try:
        try:
            with tarfile.open(filename, 'r:gz') as tar:
                tar.extractall(path=output_dir)
        except tarfile.ReadError:
            # Try lz4 compressed tar
            print("[Download] Trying lz4 decompression...")
            with lz4.frame.open(filename, 'rb') as lz4_file:
                with tarfile.open(fileobj=lz4_file, mode='r|') as tar:
                    tar.extractall(path=output_dir)
        print(f"[Download] Successfully extracted to {output_dir}")
        logger.info(f"Successfully extracted file to {output_dir}")
    except Exception as e:
        logger.error(f"Failed to extract downloaded data: {str(e)}")
        raise RuntimeError(f"Failed to extract data: {e}")

    print(f"[Download] Download and extraction completed. Data in: {output_dir}")
    return output_dir


def main():
    parser = argparse.ArgumentParser(description="Download VR dataset tar.gz files from cloud")
    parser.add_argument("--datacloud_endpoint", type=str, default="http://10.128.1.42:8083", help="Datacloud endpoint")
    parser.add_argument("--identifier", type=str, default="blockq3", help="Dataset identifier")
    parser.add_argument("--query_filter", type=str, default="{}", help="Query filter as JSON string")
    parser.add_argument("--output_dir", type=str, default=".cache/targz", help="Output directory for downloaded data")

    args = parser.parse_args()

    print("[Download] Starting VR dataset tar.gz download")
    print(f"[Download] Parameters: endpoint={args.datacloud_endpoint}, identifier={args.identifier}, output_dir={args.output_dir}")

    try:
        output_dir = download_vr_dataset_tar_gz(
            datacloud_endpoint=args.datacloud_endpoint,
            identifier=args.identifier,
            query_filter=json.loads(args.query_filter) if args.query_filter else {},
            output_dir=args.output_dir
        )
        print(f"[Download] SUCCESS: Data downloaded and extracted to {output_dir}")
        print("[Download] You can now use post_process_scripts/post_process_data_vr.py to convert to zarr")

    except Exception as e:
        print(f"[Download] FAILED: {e}")
        import traceback
        traceback.print_exc()
        exit(1)


if __name__ == "__main__":
    main()