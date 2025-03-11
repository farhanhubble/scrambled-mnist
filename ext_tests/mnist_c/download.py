import os
import sys
import urllib.request
from config import config
from tqdm import tqdm


def download_mnist_c():
    # Get environment variables or exit with an error message
    MAIN_DATASET = config.mnist_c_main_dataset
    AUX_DATASET = config.mnist_c_aux_dataset

    # Directory for external data
    external_data_dir = os.path.join(config.data_dir, config.external_data_subdir)
    os.makedirs(external_data_dir, exist_ok=True)

    # Download the main dataset
    main_zip_path = os.path.join(config.tmp_dir, "mnist_c.zip")
    _download_file(MAIN_DATASET, main_zip_path)

    # Download the auxiliary dataset
    aux_zip_path = os.path.join(config.tmp_dir, "mnist_c_leftovers.zip")
    _download_file(AUX_DATASET, aux_zip_path)


def _download_file(url, output_zip):
    """Download a file from a URL."""

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
        print("Download complete!")
    except Exception as e:
        sys.exit(f"Error downloading {url}: {e}")



