from config import config
import os
import sys
import zipfile


def _extract_zip(zip_path, extract_to):
    """Extract a zip file to a directory."""
    try:
        print(f"Extracting {zip_path}...")
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(extract_to)
        print(f"Removing {zip_path}...")
        print("Extraction complete!")
    except Exception as e:
        sys.exit(f"Error extracting {zip_path}: {e}")


def extract_mnist_c():

    # Directory for external data
    extraction_dir = os.path.join(config.data_dir, config.external_data_subdir)
    os.makedirs(extraction_dir, exist_ok=True)


    # Extract the main dataset
    main_zip_path = os.path.join(config.tmp_dir, "mnist_c.zip")
    _extract_zip(main_zip_path, extraction_dir)

    # Extract the auxiliary dataset
    aux_zip_path = os.path.join(config.tmp_dir, "mnist_c_leftovers.zip")
    _extract_zip(aux_zip_path, extraction_dir)
