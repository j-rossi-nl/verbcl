import json
import glob
import logging
import os
import pyarrow.dataset as ds
import pyarrow.parquet as pq
import random
import shutil
import string
import tqdm

from pymongo.command_cursor import CommandCursor
from pymongo.cursor import Cursor
from typing import Any, Callable, Dict, Iterator, Iterable, List, Optional, Tuple, Union


def parquet_dataset_iterator(dataset: ds.FileSystemDataset, batch_size: Optional[int] = None) -> \
        Tuple[Callable[[], Iterator[Any]], int]:
    def _iterator():
        for scan_task in dataset.scan(batch_size=1024 if batch_size is None else batch_size):
            for batch in scan_task.execute():
                yield batch

    nb_rows = sum(pq.read_metadata(f).num_rows for f in dataset.files)
    return _iterator, nb_rows


def list_iterator(items: List[Any]) -> Tuple[Callable[[], Iterator[Any]], int]:
    def _iterator():
        yield from items

    return _iterator, len(items)


def batch_iterator_from_sliceable(items: Any, batch_size: int) -> Callable[[], Iterator[Any]]:
    def _iterator():
        batch_id = 0
        while True:
            logging.info("Get one batch")
            batch = items[batch_id * batch_size: (batch_id + 1) * batch_size]
            if len(batch) > 0:
                yield batch
                if len(batch) < batch_size:
                    break
            batch_id += 1

    return _iterator


def batch_iterator_from_iterable(items: Iterable[Any], batch_size: int) -> Callable[[], Iterator[Any]]:
    def _iterator():
        while True:
            logging.info("Get one batch")
            batch = [x for _, x in zip(range(batch_size), items)]
            yield batch
            if len(batch) < batch_size:
                break

    return _iterator


def batch_iterator(items: List[Any], batch_size: int) -> Tuple[Callable[[], Iterator[Any]], int]:
    list_iter_fn, _ = list_iterator(items)
    list_iter = list_iter_fn()

    def _iterator():
        while True:
            batch = [x for _, x in zip(range(batch_size), list_iter)]
            yield batch
            if len(batch) < batch_size:
                break

    return _iterator, len(items)


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


def write_jsonl(data: List[Dict[Any, Any]], path: str, append: bool = False) -> None:
    mode = "a" if append else "w"
    with open(path, mode) as out:
        for d in data:
            out.write(json.dumps(d) + "\n")


def read_jsonl(path: str) -> List[Any]:
    with open(path) as src:
        data = [json.loads(line) for line in src]

    return data
