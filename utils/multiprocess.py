import functools
import logging
import multiprocessing
import queue
import time
import tqdm

from multiprocessing import current_process, Pool
from threading import Thread
from typing import Any, Callable, Iterator, NewType, Union

# Define new type for type hinting
Queue = NewType("Queue", Union[multiprocessing.Queue, queue.Queue])
QueueWorker = NewType("QueueWorker", Callable[[Queue, Queue], None])
Work = NewType("Work", Callable[[Any], int])


def queue_worker(func: Work) -> QueueWorker:
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
            logging.debug(f'{current_process()}: Wait in_queue')
            x = in_queue.get(True)
            logging.debug(f'{current_process()}: Got in_queue')
            start = time.time()
            out = func(x)
            end = time.time()
            logging.debug(f'{current_process()}: Finished batch in {end - start:.1f}s')
            out_queue.put(out, block=True, timeout=1)
            logging.debug(f'{current_process()}: result in out_queue')

    return wrapper


def multiprocess(worker_fn: QueueWorker,
                 input_iterator_fn: Callable[[], Iterator[Any]],
                 total: int,
                 nb_workers: int,
                 description: str = 'Process') -> None:
    """
    BASED ON MULTIPROCESSING. Any underlying shared object should be resistant to fork().
    Another kind of multiprocessing with progress bar. It uses Queue to keep track of progress.
    The worker is implemented in worker_fn, a function that receives 2 queues, an in_queue and an out_queue.
    The expected behavior of the worker is: read from in_queue, process the data, put it in out_queue (1 for 1)
    See @queue_worker

    :param worker_fn: the worker function. It gets an in_queue and and out_queue.
    :param input_iterator_fn: an iterator over input data for the process
    :param total: indicates what is the end result when summing all returned values of the decorated worker
    :param nb_workers: Number of parallel workers
    :param description: The description string for the progress bar
    """

    # The parent process will  feed data to a 'to do' queue, the workers will read from this queue and feed a 'done'
    # queue that the parent process reads
    todo_queue = multiprocessing.Queue(128)  # Not too big to avoid memory overload
    done_queue = multiprocessing.Queue()

    # This is the parent process
    # Launch the workers
    pool = Pool(processes=nb_workers,
                initializer=worker_fn,
                initargs=(todo_queue, done_queue))

    # Fill in the "to do" queue in a separate thread
    def _fill_todo_queue():
        for x in input_iterator_fn():
            logging.debug(f'Adding to IN-Queue')
            todo_queue.put(x)
            logging.debug(f' added to IN-Queue')
        logging.debug(f'All samples in IN-Queue')

    fillin = Thread(target=_fill_todo_queue)
    fillin.start()

    # Use the "done" queue to monitor process
    with tqdm.tqdm(desc=description, total=total, smoothing=0.1) as pbar:
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


def multithread(worker_fn: QueueWorker,
                input_iterator_fn: Callable[[], Iterator[Any]],
                total: int,
                nb_workers: int,
                description: str = 'Process') -> None:
    """
    BASED ON MULTITHREADING. No fork() will be performed, the shared objects have to be thread-safe.
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
    todo_queue = queue.Queue(1024)  # Not too big to avoid memory overload
    done_queue = queue.Queue()

    # Fill in the "to do" queue in a separate thread
    def _fill_todo_queue():
        for x in input_iterator_fn():
            logging.debug(f'Adding to IN-Queue')
            todo_queue.put(x)
            logging.debug(f' added to IN-Queue')
        logging.debug(f'All samples in IN-Queue')

    fillin = Thread(target=_fill_todo_queue, daemon=True)
    fillin.start()

    # This is the parent process
    # Launch the workers
    threads = [Thread(target=worker_fn, args=(todo_queue, done_queue), daemon=True) for _ in range(nb_workers)]
    for t in threads:
        t.start()

    # Use the "done" queue to monitor process
    with tqdm.tqdm(desc=description, total=total, smoothing=0.1) as pbar:
        while pbar.n < pbar.total:
            results: int = done_queue.get(True)
            pbar.update(n=results)

    # At this point, all threads are just blocked waiting to read something from the now empty 'to do' queue.
    assert todo_queue.empty()
    assert done_queue.empty()
