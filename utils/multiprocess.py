import tqdm
import threading
import functools

from multiprocessing import Pool, Queue
from typing import Callable, Iterator, Any


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
