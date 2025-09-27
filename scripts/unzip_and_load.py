# scripts/unzip_and_load.py

import zipfile
from pathlib import Path
from src.ingestion.data_loader import load_json_tickets

def unzip_dataset(zip_path: str, extract_to: str = "data/") -> str:
    """Unzip the dataset and return extracted file path."""
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(extract_to)
        files = zip_ref.namelist()
    print(f"✅ Extracted files: {files}")
    return str(Path(extract_to) / files[0])  # assume single file

if __name__ == "__main__":
    zip_path = "support_tickets.zip"
    output_dir = "data/"
    json_file = unzip_dataset(zip_path, output_dir)

    # Load & check data
    df = load_json_tickets(json_file)
    print(f"Loaded {df.shape[0]} tickets with {df.shape[1]} fields")
    print(df.head(2))
