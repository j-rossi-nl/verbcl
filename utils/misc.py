import random
import string
import pyarrow.dataset as ds
import pyarrow.parquet as pq

from typing import Callable, Iterator, Any, List, Tuple, Optional


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


def random_name(n=6):
    return ''.join((random.choice(string.ascii_letters + string.digits) for _ in range(n)))
