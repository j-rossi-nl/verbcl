import tqdm
import threading
import random
import string
import pyarrow.dataset as ds
import pyarrow.parquet as pq

from multiprocessing import Pool, Queue
from typing import Callable, Iterator, Any, List


def multiprocess_with_queue(worker_fn: Callable[[Queue, Queue], None],
                            input_iterator_fn: Callable[[], Iterator[Any]],
                            nb_workers: int,
                            description: str = 'Process') -> List[Any]:
    """
    Another kind of multiprocessing with progress bar. It uses Queue to keep track of progress.
    The worker is implemented in worker_fn, a function that receives 2 queues, an in_queue and an out_queue. The
    expected behavior of the worker is: read from in_queue, process the data, put it in out_queue (1 for 1)

    :param worker_fn: the worker function. It gets an in_queue and and out_queue.
    :param input_iterator_fn: an iterator over input data for the process
    :param nb_workers: numbers of workers to spawn
    :param description: The description string for the progress bar
    :return: list of results
    """
    # To change a bit from the monitoring system in utils, we implement a queue-based feeder-consumer model
    # The parent process will  feed data to a 'to do' queue, the workers will read from this queue and feed a 'done'
    # queue that the parent process reads
    todo_queue = Queue()
    done_queue = Queue()

    # Fill in the "to do" queue
    for x in input_iterator_fn():
        todo_queue.put(x)

    # Safe to use, as the workers have not started yet
    nb_inputs = todo_queue.qsize()

    # This is the parent process
    # Launch the workers
    pool = Pool(processes=nb_workers,
                initializer=worker_fn,
                initargs=(todo_queue, done_queue))

    # Use the "done" queue to monitor process
    collected_outputs = []
    with tqdm.tqdm(desc=description, total=nb_inputs) as pbar:
        # We know how many items we expect
        for _ in range(nb_inputs):
            results = done_queue.get(True)
            if results is not None:
                collected_outputs.append(results)
            pbar.update()

    # At this point, all threads are just blocked waiting to read something from the now empty 'to do' queue.
    assert todo_queue.empty()
    assert done_queue.empty()

    # Let's terminate the workers
    pool.terminate()
    pool.join()

    return collected_outputs


def multiprocess_dataset(worker_fn: Callable[[Queue, Queue], None],
                         input_dataset_path: str,
                         nb_workers: int,
                         description: str = 'Process') -> None:
    """
    Another kind of multiprocessing with progress bar. It uses Queue to keep track of progress.
    It goes over a full pyarrrow dataset, ie a folder with Parquet files
    The worker is implemented in worker_fn, a function that receives 2 queues, an in_queue and an out_queue. The
    expected behavior of the worker is: read from in_queue, process the data, put it in out_queue (1 for 1)

    :param worker_fn: the worker function. It gets an in_queue and and out_queue.
    :param input_dataset_path: the dataset to process (path to the dataset folder_
    :param nb_workers: numbers of workers to spawn
    :param description: The description string for the progress bar
    """

    # The parent process will  feed data to a 'to do' queue, the workers will read from this queue and feed a 'done'
    # queue that the parent process reads
    todo_queue = Queue(128)   # Not too big to avoid memory overload
    done_queue = Queue()

    dataset: ds.Dataset = ds.dataset(input_dataset_path)
    dataset: ds.FileSystemDataset  # We know the actual subclass
    nb_rows = sum(pq.read_metadata(f).num_rows for f in dataset.files)

    # This is the parent process
    # Launch the workers
    pool = Pool(processes=nb_workers,
                initializer=worker_fn,
                initargs=(todo_queue, done_queue))

    # Fill in the "to do" queue
    def _fill_todo_queue():
        for scan_task in dataset.scan(batch_size=1024):
            for batch in scan_task.execute():
                todo_queue.put(batch)  # Block if queue is full until a slot is free (no timeout)

    fillin = threading.Thread(target=_fill_todo_queue)
    fillin.start()

    # Use the "done" queue to monitor process
    with tqdm.tqdm(desc=description, total=nb_rows) as pbar:
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


def random_name(n=6):
    return ''.join((random.choice(string.ascii_letters + string.digits) for _ in range(n)))


