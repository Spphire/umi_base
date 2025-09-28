#!/usr/bin/env python3
"""
Simple script to download data by identifier to ignore-data directory
Usage: python download_data.py <identifier>
Example: python download_data.py "New pp"
"""

import sys
import os
import requests
import shutil
from tqdm import tqdm

def download_data(identifier, endpoint="http://127.0.0.1:8083"):
    """Download data for given identifier to ignore-data directory"""

    # Create output directory
    output_dir = os.path.join("ignore-data2", identifier.replace(" ", "_"))
    os.makedirs(output_dir, exist_ok=True)

    print(f"Downloading data for identifier: '{identifier}'")

    # Step 1: Get records list
    try:
        response = requests.get(f"{endpoint}/v1/logs/{identifier}")
        records = response.json().get('data', [])
        if not records:
            print(f"No records found for identifier: '{identifier}'")
            return

        print(f"Found {len(records)} records")
        uuid_list = [record['uuid'] for record in records]

    except Exception as e:
        print(f"Failed to get records: {e}")
        return

    # Step 2: Download data
    try:
        data_request = {
            "identifier": identifier,
            "uuids": uuid_list,
        }

        response = requests.post(
            f"{endpoint}/v1/download_records",
            json=data_request,
            stream=True
        )

        if response.status_code != 200:
            print(f"Download failed: {response.text}")
            return

        # Download with progress bar
        filename = os.path.join(output_dir, "data.tar.gz")
        total_size = int(response.headers.get('content-length', 0))

        with open(filename, 'wb') as f:
            with tqdm(total=total_size, unit='B', unit_scale=True, desc="Downloading") as pbar:
                for chunk in tqdm(response.iter_content(chunk_size=1024*1024)):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))

        print(f"Downloaded to: {filename}")

        # Step 3: Extract
        print("Extracting...")
        shutil.unpack_archive(filename, output_dir, 'gztar')
        os.remove(filename)  # Remove tar.gz after extraction

        print(f"Data extracted to: {output_dir}")

    except Exception as e:
        print(f"Download failed: {e}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python download_data.py <identifier>")
        print('Example: python download_data.py "New pp"')
        sys.exit(1)

    identifier = sys.argv[1]
    download_data(identifier)
