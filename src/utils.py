import random
import string
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq

from typing import Callable, Iterator, Any, List, Tuple, Optional

from opinion import Opinion


def parquet_dataset_iterator(dataset: ds.FileSystemDataset, batch_size: Optional[int] = None) -> Tuple[Callable[[], Iterator[Any]], int]:
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


def opinions_in_arrowbatch(x: pa.RecordBatch) -> Iterator[Opinion]:
    """
    Helper to iterate through batches of opinion records coming from PARQUET dataset of opinions.

    :param x: batch
    :return: Opinion objects
    """
    d = x.to_pydict()
    for citing_opinion_id, opinion_html in zip(d['opinion_id'], d['html_with_citations']):
        yield Opinion(opinion_id=citing_opinion_id, opinion_html=opinion_html)


def text_before_after(txt: str, span: Tuple[int, int], nb_words: int) -> Tuple[str, int, int]:
    """
    Given a text, and span within this text, extract a number of words before and after the span.
    The returned text includes the span within the original text, surrounded by nb_words.

    :param txt: original text
    :param span: a 2-uple (start, end) indicating the span of text to be preserved
    :param nb_words: how many words to extract
    :return: a snippet of text, length of text before the original span, length of text after the original span
    """
    start, end = span
    before_txt = txt[:start]
    span_txt = txt[start:end]
    after_txt = txt[end:]

    before_txt = ' '.join(before_txt.split(' ')[-nb_words:])
    after_txt = ' '.join(after_txt.split(' ')[:nb_words])

    total_txt = ''.join([before_txt, span_txt, after_txt])
    return total_txt, len(before_txt), len(after_txt)
