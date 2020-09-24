import random
import string
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq

from typing import Callable, Iterator, Any, List, Tuple

from opinion import Opinion


def parquet_dataset_iterator(dataset: ds.FileSystemDataset) -> Tuple[Callable[[], Iterator[Any]], int]:
    def _iterator():
        for scan_task in dataset.scan(batch_size=1024):
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


def opinions_in_arrowbatch(x: pa.RecordBatch) -> Iterator[Opinion]:
    """
    Helper to iterate through batches of opinion records coming from PARQUET dataset of opinions.

    :param x: batch
    :return: Opinion objects
    """
    d = x.to_pydict()
    for citing_opinion_id, opinion_html in zip(d['opinion_id'], d['html_with_citations']):
        yield Opinion(opinion_id=citing_opinion_id, opinion_html=opinion_html)
