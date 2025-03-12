from config import config
import os
from test import evaluate


def _test_mnist_c(data_root: str):
    subdirs = sorted(os.listdir(data_root))
    for subdir in subdirs:
        if os.path.isdir(os.path.join(data_root, subdir)):
            test_data_path = os.path.join(data_root, subdir, "test")
            evaluate(test_data_path)


def test_mnist_c(top_dir: str):
    _test_mnist_c(
        top_dir + "/mnist_c",
    )
    _test_mnist_c(
        top_dir + "/mnist_c_leftovers",
    )
