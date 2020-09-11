import tqdm
import threading
import random
import string
import functools
import pyarrow as pa
import bs4
import json

import pyarrow.dataset as ds
import pyarrow.parquet as pq
from multiprocessing import Pool, Queue
from typing import Callable, Iterator, Any, List, Tuple, Dict
from bs4 import BeautifulSoup

CITATION_TAG = 'CITATION'


class Opinion:
    """
    A helper class to manipulate an opinion.
    """
    bs4_citation_args = {'class': 'citation', 'data-id': True}

    def __init__(self, opinion_id: int, opinion_html: str):
        self.opinion_id = opinion_id
        self.opinion_html = opinion_html
        self.soup = clean_html(BeautifulSoup(self.opinion_html, 'html.parser'))
        self.raw_text = self.soup.get_text()
        self.num_words = len(self.raw_text.split())

    def citations(self, return_tag: bool = False) -> Iterator[Dict[str, Any]]:
        """
        An iterator to go through all the citations in the opinion. Each iteration will yield a dict with the keys
        citing_opinion_id, cited_opinion_id. If return_tag is True, then the dict also contains the key
        'tag', the value is a Tag object.
        :return: iterator
        """
        spans = self.soup.find_all('span', **Opinion.bs4_citation_args)
        for s in spans:
            data = {'citing_opinion_id': self.opinion_id, 'cited_opinion_id': int(s['data-id'])}
            if return_tag:
                data['tag'] = s
            yield data

    def __len__(self):
        return self.num_words


def queue_worker(func):
    """
    Decorator for multiprocess workers.

    :param func: signature (in: Any) -> int: processes one item, return one integer.
    :return: a worker function, with signature (in_queue: Queue, out_queue: Queue) -> None
    """

    @functools.wraps(func)
    def wrapper(in_queue: Queue, out_queue: Queue):
        """

        :param in_queue: tasks to do
        :param out_queue: done tasks
        :return:
        """
        while True:
            x = in_queue.get(True)
            out = func(x)
            out_queue.put(out)

    return wrapper


def multiprocess(worker_fn: Callable[[Queue, Queue], None],
                 input_iterator_fn: Callable[[], Iterator[Any]],
                 total: int,
                 nb_workers: int,
                 description: str = 'Process') -> None:
    """
    Another kind of multiprocessing with progress bar. It uses Queue to keep track of progress.
    It goes over a full pyarrrow dataset, ie a folder with Parquet files
    The worker is implemented in worker_fn, a function that receives 2 queues, an in_queue and an out_queue. The
    expected behavior of the worker is: read from in_queue, process the data, put it in out_queue (1 for 1)

    :param worker_fn: the worker function. It gets an in_queue and and out_queue.
    :param input_iterator_fn: an iterator over input data for the process
    :param total: indicates what is the end result when summing all returned values of the decorated worker
    :param nb_workers: Number of parallel workers
    :param description: The description string for the progress bar
    """

    # The parent process will  feed data to a 'to do' queue, the workers will read from this queue and feed a 'done'
    # queue that the parent process reads
    todo_queue = Queue(128)  # Not too big to avoid memory overload
    done_queue = Queue()

    # This is the parent process
    # Launch the workers
    pool = Pool(processes=nb_workers,
                initializer=worker_fn,
                initargs=(todo_queue, done_queue))

    # Fill in the "to do" queue in a separate thread
    def _fill_todo_queue():
        for x in input_iterator_fn():
            todo_queue.put(x)

    fillin = threading.Thread(target=_fill_todo_queue)
    fillin.start()

    # Use the "done" queue to monitor process
    with tqdm.tqdm(desc=description, total=total) as pbar:
        while pbar.n < pbar.total:
            results: int = done_queue.get(True)
            pbar.update(n=results)

    # At this point, all threads are just blocked waiting to read something from the now empty 'to do' queue.
    assert todo_queue.empty()
    assert done_queue.empty()

    # Let's terminate the workers
    pool.terminate()
    pool.join()
    fillin.join()


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


def clean_html(html: BeautifulSoup) -> BeautifulSoup:
    # A bit of cleaning on the original HTML
    # It includes tags for original pagination that insert numbers here and there in the text
    bs4_pagination_args = {'class': 'pagination'}
    bs4_citation_args_nolink = {'class': 'citation no-link'}

    for page in html.find_all('span', **bs4_pagination_args):
        page: bs4.element.Tag
        page.decompose()

    for nolink in html.find_all('span', **bs4_citation_args_nolink):
        nolink: bs4.element.Tag
        nolink.decompose()

    return html


def opinions_in_arrowbatch(x: pa.RecordBatch) -> Iterator[Opinion]:
    """
    Helper to iterate through batches of opinion records coming from PARQUET dataset of opinions.

    :param x: batch
    :return: Opinion objects
    """
    d = x.to_pydict()
    for citing_opinion_id, opinion_html in zip(d['opinion_id'], d['html_with_citations']):
        yield Opinion(opinion_id=citing_opinion_id, opinion_html=opinion_html)


def citation_to_jsonl(x: bs4.Tag, max_words_before_after: int) -> str:
    """
    From a citation in an opinion, generate the snippet of text that will be used by annotators.
    It is returned in JSONL format, and the citation itself is marked as 'CITATION'.
    It is assumed the given tag is actually a citation from an opinion.

    :param x:  a tag 'citation' in an opinion
    :return: JSONL string
    """

    def concatenate_txts(tags):
        def _tags_iter():
            for s in tags:
                # s will be either a NavigableString or a Tag
                if isinstance(s, bs4.NavigableString):
                    s: bs4.NavigableString
                    s_txt = str(s)
                elif isinstance(s, bs4.Tag):
                    s: bs4.Tag
                    s_txt = s.get_text()
                else:
                    # Well well, what do we have here??
                    s_txt = ''
                yield s_txt

        return ''.join(t for t in _tags_iter())

    txts = list(map(concatenate_txts, [x.previous_siblings, [x], x.next_siblings]))
    before_citation_txt, citation_txt, after_citation_txt = txts

    # Limit to a maximum number of characters before and after the citation
    before_citation_txt = ' '.join(before_citation_txt.split(' ')[-max_words_before_after:]).replace('\n', ' ').strip()
    after_citation_txt = ' '.join(after_citation_txt.split(' ')[:max_words_before_after]).replace('\n', ' ').strip()

    start_citation = len(before_citation_txt)
    end_citation = start_citation + len(citation_txt)

    full_text = ''.join([before_citation_txt, citation_txt, after_citation_txt])
    data = {'text': full_text, 'labels': [[start_citation, end_citation, CITATION_TAG]]}
    return json.dumps(data)
