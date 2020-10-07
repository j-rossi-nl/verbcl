import sys
import os
import logging
import logging.config
import functools
import elasticsearch
import pyarrow as pa
import pyarrow.dataset as ds

import config

from elasticsearch import Elasticsearch
from argparse import ArgumentParser, Namespace
from dotenv import load_dotenv
from typing import Any

from opinion_dataset import OpinionDataset
from opinion import Opinion
from multiprocess import multiprocess, queue_worker
from utils import opinions_in_arrowbatch, parquet_dataset_iterator

config.config()
elasticsearch.logger.setLevel(logging.WARNING)
_args: Namespace = Namespace()


def elastic_function(func):
    """
    Decorator for a function connecting to Elastic Search.

    :param func: signature (connection: Elasticsearch) -> None: uses a connection
    :return: a function with no argument
    """

    @functools.wraps(func)
    def wrapper():
        """

        :return:
        """
        load_dotenv(_args.env)
        connection = Elasticsearch(cloud_id=os.getenv('CLOUD_ID'), api_key=(os.getenv("ID"), os.getenv("KEY")))
        func(connection)

    return wrapper


@elastic_function
def index_all_opinions(es: Elasticsearch):
    """
    Adds all opinions to an index.

    :return:
    """
    @queue_worker
    def _send_request(x: pa.RecordBatch) -> int:
        for opinion in opinions_in_arrowbatch(x):
            es.index(
                index=_args.index,
                body={
                    'opinion_id': opinion.opinion_id,
                    'raw_text': opinion.raw_text
                }
            )
        return x.num_rows

    logging.info(f'Processing the dataset in {_args.path}')
    dataset: Any = ds.dataset(_args.path)
    dataset: ds.FileSystemDataset
    iterator, nb_rows = parquet_dataset_iterator(dataset=dataset, batch_size=128)
    multiprocess(worker_fn=_send_request, input_iterator_fn=iterator, total=nb_rows,
                 nb_workers=_args.num_workers, description=f'Add opinions to index {_args.index}')


def parse_args():
    argstxt = sys.argv[1:]
    parser = ArgumentParser()
    subparsers = parser.add_subparsers(title='Subcommands', description='Valid subcommands',
                                       help='Additional help')

    parser_index = subparsers.add_parser(name='index', description='Add all opinions to the index.')
    parser_index.add_argument('--path', type=str, help='Path to the PARQUET dataset')
    parser_index.add_argument('--index', type=str, help='Index name')
    parser_index.add_argument('--env', type=str, help='Path to the env file')
    parser_index.add_argument('--num-workers', type=int, default=2, help='Parallel workers.')
    parser_index.set_defaults(func=index_all_opinions)
    return parser.parse_args(argstxt)


def main():
    global _args
    _args = parse_args()
    _args.func()
    logging.info('Done.')


if __name__ == '__main__':
    main()
