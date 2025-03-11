import os
import sys
import urllib.request
import zipfile
from config import config
from tqdm import tqdm


def download_mnist_c():
    # Get environment variables or exit with an error message
    MAIN_DATASET = config.mnist_c_main_dataset
    AUX_DATASET = config.mnist_c_aux_dataset

    # Directory for external data
    external_data_dir = "data/external/raw"
    os.makedirs(external_data_dir, exist_ok=True)

    # Download and extract the main dataset
    _download_and_extract(
        MAIN_DATASET, f"{external_data_dir}/mnist_c.zip", external_data_dir
    )

    # Download and extract the auxiliary dataset
    _download_and_extract(
        AUX_DATASET, f"{external_data_dir}/mnist_c_leftovers.zip", external_data_dir
    )


def _download_and_extract(url, output_zip, extract_to):
    """Download a file from a URL and extract it to a directory."""

    def _progress_hook(count, block_size, total_size):
        if _progress_hook.tqdm_obj is None:
            _progress_hook.tqdm_obj = tqdm(
                total=total_size,
                unit="B",
                unit_scale=True,
                desc="Downloading",
                ascii=True,
            )
        _progress_hook.tqdm_obj.update(block_size)

    try:
        print(f"Downloading {url}...")
        _progress_hook.tqdm_obj = None
        urllib.request.urlretrieve(url, output_zip, _progress_hook)
        if _progress_hook.tqdm_obj:
            _progress_hook.tqdm_obj.close()

        print(f"Extracting {output_zip}...")
        with zipfile.ZipFile(output_zip, "r") as zip_ref:
            zip_ref.extractall(extract_to)

        print(f"Removing {output_zip}...")
        os.remove(output_zip)
        print("Done!")
    except Exception as e:
        sys.exit(f"Error processing {url}: {e}")
