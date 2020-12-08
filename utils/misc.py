import os
import pyarrow.dataset as ds
import pyarrow.parquet as pq
import random
import shutil
import string

from pymongo.command_cursor import CommandCursor
from pymongo.cursor import Cursor
from typing import Callable, Iterator, Any, List, Optional, Tuple, Union


def parquet_dataset_iterator(dataset: ds.FileSystemDataset, batch_size: Optional[int] = None) -> \
        Tuple[Callable[[], Iterator[Any]], int]:
    def _iterator():
        for scan_task in dataset.scan(batch_size=1024 if batch_size is None else batch_size):
            for batch in scan_task.execute():
                yield batch

    nb_rows = sum(pq.read_metadata(f).num_rows for f in dataset.files)
    return _iterator, nb_rows


def file_list_iterator(files: List[str]) -> Tuple[Callable[[], Iterator[Any]], int]:
    def _iterator():
        for f in files:
            yield f

    return _iterator, len(files)


def mongodb_cursor_iterator(cursor: Union[CommandCursor, Cursor], batch_size: Optional[int] = None) -> \
        Callable[[], Iterator[Any]]:
    batch_size = batch_size if batch_size is not None else 1024

    def _iterator():
        while True:
            batch = [d for _, d in zip(range(batch_size), cursor)]
            if len(batch) == 0:
                break
            yield batch

    return _iterator


def random_name(n=6):
    return ''.join((random.choice(string.ascii_letters + string.digits) for _ in range(n)))


def make_clean_folder(path):
    """
    Create the folder, or clean if it does already exist.

    :param path: path to a folder
    :return:
    """
    if os.path.exists(path):
        shutil.rmtree(path)
        os.makedirs(path, exist_ok=True)
