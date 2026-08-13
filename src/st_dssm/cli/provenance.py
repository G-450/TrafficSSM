import argparse
import json
import os
import sys
import urllib.request

import yaml

from st_dssm.data import ZENODO_URLS, DataProvenanceError, verify_dataset


def download_file(url: str, dest_path: str):
    print(f"Downloading {url} to {dest_path}...")
    temp_path = dest_path + ".tmp"
    try:
        urllib.request.urlretrieve(url, temp_path)
        os.replace(temp_path, dest_path)
    except Exception:
        if os.path.exists(temp_path):
            os.remove(temp_path)
        raise

def main():
    parser = argparse.ArgumentParser(description="ST-DSSM Data Provenance Tool")
    parser.add_argument("--config", type=str, default="configs/provenance.yaml", help="Path to config file")
    parser.add_argument("--download", action="store_true", help="Download missing files from Zenodo")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    
    data_dir = config.get("data_dir", "data/raw")
    manifest_path = config.get("manifest_path", "data/manifest/pems_bay_manifest.json")

    os.makedirs(data_dir, exist_ok=True)
    
    if args.download:
        for filename, url in ZENODO_URLS.items():
            dest_path = os.path.join(data_dir, filename)
            if not os.path.exists(dest_path):
                download_file(url, dest_path)
            else:
                print(f"File already exists: {dest_path}")
                
    try:
        print("Verifying dataset...")
        manifest = verify_dataset(data_dir)
        print("Verification PASSED.")
        
        os.makedirs(os.path.dirname(manifest_path), exist_ok=True)
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)
        print(f"Manifest written to {manifest_path}")
        sys.exit(0)
    except DataProvenanceError as e:
        print(f"Verification FAILED: {e}", file=sys.stderr)
        sys.exit(1)
    except FileNotFoundError as e:
        print(f"Verification FAILED: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:  # noqa: BLE001
        print(f"Unexpected error: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
